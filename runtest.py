#!/usr/bin/env python3
"""
Benchmark test comparing registration speed between LaMAR and direct ANTs registration.

This script runs registration on the same image pair using both:
1. LaMAR's parcellation-based approach (via SynthSeg)
2. Direct ANTs registration using the same parameters

It measures and reports the execution time for both methods.
"""

import os
import sys
import time
import shutil
import tempfile
import argparse
import subprocess
import nibabel as nib
import numpy as np
import torch


def run_lamar_registration(
    moving_img,
    fixed_img,
    output_dir,
    registration_method="SyNRA",
    threads=1,
    verbose=True,
):
    """Run registration using LaMAR CLI and measure time."""
    start_time = time.time()

    # Set up output paths
    output_img = os.path.join(output_dir, "lamar_registered.nii.gz")
    moving_parc = os.path.join(output_dir, "moving_parc.nii.gz")
    fixed_parc = os.path.join(output_dir, "fixed_parc.nii.gz")
    registered_parc = os.path.join(output_dir, "registered_parc.nii.gz")
    affine_file = os.path.join(output_dir, "lamar_affine.mat")
    warp_file = os.path.join(output_dir, "lamar_warp.nii.gz")
    # inverse_warp = os.path.join(output_dir, "lamar_inverse_warp.nii.gz")
    # inverse_affine = os.path.join(output_dir, "lamar_inverse_affine.mat")

    # Build command for lamar registration
    cmd = [
        "lamar",
        "--moving",
        moving_img,
        "--fixed",
        fixed_img,
        "--output",
        output_img,
        "--moving-parc",
        moving_parc,
        "--fixed-parc",
        fixed_parc,
        "--registered-parc",
        registered_parc,
        "--affine",
        affine_file,
        "--warpfield",
        warp_file,
        # "--inverse-warpfield",
        # inverse_warp,
        # "--inverse-affine",
        # inverse_affine,
        "--registration-method",
        registration_method,
        "--synthseg-threads",
        str(threads),
        "--ants-threads",
        str(threads),
        "--skip-qc",
        # "--skip-fixed-parc",
        # "--skip-moving-parc",
    ]

    # Run LaMAR registration
    if verbose:
        subprocess.run(cmd, check=True)
    else:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    elapsed_time = time.time() - start_time
    return elapsed_time, output_img


def run_direct_ants_registration(
    moving_img,
    fixed_img,
    output_dir,
    registration_method="SyNRA",
    threads=1,
    verbose=True,
):
    """Run direct ANTs registration via ANTsPyX and measure time."""
    import ants

    start_time = time.time()

    # Set up output paths
    output_img = os.path.join(output_dir, "direct_ants_registered.nii.gz")
    # affine_file = os.path.join(output_dir, "direct_ants_affine.mat")
    # warp_file = os.path.join(output_dir, "direct_ants_warp.nii.gz")
    # inverse_warp = os.path.join(output_dir, "direct_ants_inverse_warp.nii.gz")
    # inverse_affine = os.path.join(output_dir, "direct_ants_inverse_affine.mat")

    env = os.environ.copy()
    # Set ANTs/ITK thread count
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)  # OpenMP threads for ANTs

    # Load images
    fixed_image = ants.image_read(fixed_img)
    moving_image = ants.image_read(moving_img)

    # Map registration method to ANTs type
    type_of_transform = registration_method

    # Log if verbose
    if verbose:
        print(f"Running ANTsPyX registration with method: {type_of_transform}")
        print(f"Thread count: {threads}")
        print(f"Moving image: {moving_img}")
        print(f"Fixed image: {fixed_img}")

        # Perform registration
    registration = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform=type_of_transform,
        verbose=verbose,
    )

    # Save outputs
    ants.image_write(registration["warpedmovout"], output_img)

    # # Save transformation files
    # if "fwdtransforms" in registration and len(registration["fwdtransforms"]) > 0:
    #     for i, transform in enumerate(registration["fwdtransforms"]):
    #         if ".mat" in transform:
    #             shutil.copy(transform, affine_file)
    #         elif ".nii" in transform:
    #             shutil.copy(transform, warp_file)

    # if "invtransforms" in registration and len(registration["invtransforms"]) > 0:
    #     for i, transform in enumerate(registration["invtransforms"]):
    #         if ".mat" in transform:
    #             shutil.copy(transform, inverse_affine)
    #         elif ".nii" in transform:
    #             shutil.copy(transform, inverse_warp)

    elapsed_time = time.time() - start_time
    return elapsed_time, output_img


def compare_registration_quality(lamar_output, ants_output, fixed_img):
    """Compare the registration quality using all available metrics.

    Args:
        lamar_output: Path to LaMAR registered image
        ants_output: Path to ANTs registered image
        fixed_img: Path to fixed reference image

    Returns:
        Dictionary with results for all metrics
    """
    # Load images
    lamar_img_nib = nib.load(lamar_output)
    ants_img_nib = nib.load(ants_output)
    fixed_img_nib = nib.load(fixed_img)

    # Convert to numpy arrays
    lamar_img_data = lamar_img_nib.get_fdata()
    ants_img_data = ants_img_nib.get_fdata()
    fixed_img_data = fixed_img_nib.get_fdata()

    # Convert to PyTorch tensors
    lamar_tensor = torch.from_numpy(lamar_img_data).float()
    ants_tensor = torch.from_numpy(ants_img_data).float()
    fixed_tensor = torch.from_numpy(fixed_img_data).float()

    # Add batch and channel dimensions
    lamar_tensor = lamar_tensor.unsqueeze(0).unsqueeze(0)
    ants_tensor = ants_tensor.unsqueeze(0).unsqueeze(0)
    fixed_tensor = fixed_tensor.unsqueeze(0).unsqueeze(0)

    results = {}

    # Calculate NMI scores (always calculate as fallback)
    def normalized_mutual_information(img1, img2, bins=32):
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        hist_joint, _, _ = np.histogram2d(img1_flat, img2_flat, bins=bins)
        hist_img1 = np.sum(hist_joint, axis=1)
        hist_img2 = np.sum(hist_joint, axis=0)
        p_img1 = hist_img1 / np.sum(hist_img1)
        p_img2 = hist_img2 / np.sum(hist_img2)
        p_joint = hist_joint / np.sum(hist_joint)
        eps = np.finfo(float).eps
        h_img1 = -np.sum(p_img1 * np.log2(p_img1 + eps))
        h_img2 = -np.sum(p_img2 * np.log2(p_img2 + eps))
        h_joint = -np.sum(p_joint * np.log2(p_joint + eps))
        return (h_img1 + h_img2) / h_joint if h_joint > 0 else 0

    results["nmi"] = {
        "lamar": normalized_mutual_information(lamar_img_data, fixed_img_data),
        "ants": normalized_mutual_information(ants_img_data, fixed_img_data),
    }

    # Try MIND metric
    try:
        from torch_mind import MINDLoss

        mind_loss = MINDLoss(
            non_local_region_size=7,
            patch_size=5,
            neighbor_size=3,
            gaussian_patch_sigma=2.0,
        )

        with torch.no_grad():
            mid_slice = lamar_tensor.shape[2] // 2
            lamar_mind = -mind_loss(
                lamar_tensor[:, :, mid_slice, :, :], fixed_tensor[:, :, mid_slice, :, :]
            ).item()
            ants_mind = -mind_loss(
                ants_tensor[:, :, mid_slice, :, :], fixed_tensor[:, :, mid_slice, :, :]
            ).item()

        results["mind"] = {"lamar": lamar_mind, "ants": ants_mind}
    except Exception as e:
        print(f"Error calculating MIND: {e}")
        results["mind"] = None

    # Try NGF metric
    try:
        from normalized_gradient_field import NormalizedGradientField3d

        pixel_spacing = lamar_img_nib.header.get_zooms()[:3]

        ngf = NormalizedGradientField3d(
            grad_method="default",
            gauss_sigma=0.5,
            eps=1e-5,
            mm_spacing=pixel_spacing,
            reduction="mean",
        )

        with torch.no_grad():
            lamar_ngf = -ngf(lamar_tensor, fixed_tensor).item()
            ants_ngf = -ngf(ants_tensor, fixed_tensor).item()

        results["ngf"] = {"lamar": lamar_ngf, "ants": ants_ngf}
    except Exception as e:
        print(f"Error calculating NGF: {e}")
        results["ngf"] = None

    return results


def save_ngf_fields(
    source_img, target_img, output_prefix, mm_spacing=None, gauss_sigma=0.5
):
    """
    Calculate and save normalized gradient fields as NIfTI files.

    Args:
        source_img: Path to source NIfTI image or PyTorch tensor
        target_img: Path to target NIfTI image or PyTorch tensor
        output_prefix: Prefix for output filenames
        mm_spacing: Optional pixel spacing (will be extracted from header if None)
        gauss_sigma: Gaussian smoothing sigma
    """
    import nibabel as nib
    import torch
    from normalized_gradient_field import NormalizedGradientField3d
    from kernels import spatial_filter_nd
    import os

    # Handle different input types
    if isinstance(source_img, str):
        # Load images from file paths
        src_nib = nib.load(source_img)
        tgt_nib = nib.load(target_img)

        # Get spacing from header if not provided
        if mm_spacing is None:
            mm_spacing = src_nib.header.get_zooms()[:3]

        # Convert to torch tensors
        src_data = torch.from_numpy(src_nib.get_fdata()).float()
        tgt_data = torch.from_numpy(tgt_nib.get_fdata()).float()

        # Add batch and channel dimensions
        src_data = src_data.unsqueeze(0).unsqueeze(0)
        tgt_data = tgt_data.unsqueeze(0).unsqueeze(0)

        affine = src_nib.affine
        header = src_nib.header
    else:
        # Assume tensors were passed directly
        src_data = source_img
        tgt_data = target_img
        affine = np.eye(4)  # Default affine
        header = None

        if mm_spacing is None:
            mm_spacing = [1.0, 1.0, 1.0]  # Default spacing

    # Create NGF calculator
    ngf_calc = NormalizedGradientField3d(
        grad_method="default",
        gauss_sigma=gauss_sigma,
        eps=1e-5,
        mm_spacing=mm_spacing,
        reduction="none",  # Important: no reduction to keep the full field
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    # Calculate and save gradient fields
    with torch.no_grad():
        # Get source and target shape
        b, c = src_data.shape[:2]
        spatial_shape = src_data.shape[2:]

        # Reshape
        source = src_data.view(b * c, 1, *spatial_shape)
        target = tgt_data.view(b * c, 1, *spatial_shape)

        # Apply smoothing if needed
        if ngf_calc.gauss_kernel_x is not None:
            source = spatial_filter_nd(source, ngf_calc.gauss_kernel_x)
            target = spatial_filter_nd(target, ngf_calc.gauss_kernel_x)

        # Calculate gradients
        src_grad_x = (
            spatial_filter_nd(source, ngf_calc.grad_u_kernel) * ngf_calc.mm_spacing[0]
        )
        src_grad_y = (
            spatial_filter_nd(source, ngf_calc.grad_v_kernel) * ngf_calc.mm_spacing[1]
        )
        src_grad_z = (
            spatial_filter_nd(source, ngf_calc.grad_w_kernel) * ngf_calc.mm_spacing[2]
        )

        tar_grad_x = (
            spatial_filter_nd(target, ngf_calc.grad_u_kernel) * ngf_calc.mm_spacing[0]
        )
        tar_grad_y = (
            spatial_filter_nd(target, ngf_calc.grad_v_kernel) * ngf_calc.mm_spacing[1]
        )
        tar_grad_z = (
            spatial_filter_nd(target, ngf_calc.grad_w_kernel) * ngf_calc.mm_spacing[2]
        )

        # Calculate gradient magnitudes
        src_grad_norm = src_grad_x**2 + src_grad_y**2 + src_grad_z**2 + ngf_calc.eps**2
        tar_grad_norm = tar_grad_x**2 + tar_grad_y**2 + tar_grad_z**2 + ngf_calc.eps**2

        # Calculate dot product
        grad_dot_product = (
            src_grad_x * tar_grad_x + src_grad_y * tar_grad_y + src_grad_z * tar_grad_z
        )

        # Calculate NGF field
        ngf_field = -0.5 * (grad_dot_product**2 / (src_grad_norm * tar_grad_norm))
        ngf_field = ngf_field.view(b, c, *spatial_shape)

        # Convert to numpy and remove batch/channel dimensions
        src_grad_x_np = src_grad_x.view(b, c, *spatial_shape)[0, 0].cpu().numpy()
        src_grad_y_np = src_grad_y.view(b, c, *spatial_shape)[0, 0].cpu().numpy()
        src_grad_z_np = src_grad_z.view(b, c, *spatial_shape)[0, 0].cpu().numpy()

        tar_grad_x_np = tar_grad_x.view(b, c, *spatial_shape)[0, 0].cpu().numpy()
        tar_grad_y_np = tar_grad_y.view(b, c, *spatial_shape)[0, 0].cpu().numpy()
        tar_grad_z_np = tar_grad_z.view(b, c, *spatial_shape)[0, 0].cpu().numpy()

        ngf_field_np = ngf_field[0, 0].cpu().numpy()

        # Create NIfTI images
        nib.save(
            nib.Nifti1Image(src_grad_x_np, affine, header),
            f"{output_prefix}_src_grad_x.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(src_grad_y_np, affine, header),
            f"{output_prefix}_src_grad_y.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(src_grad_z_np, affine, header),
            f"{output_prefix}_src_grad_z.nii.gz",
        )

        nib.save(
            nib.Nifti1Image(tar_grad_x_np, affine, header),
            f"{output_prefix}_tar_grad_x.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(tar_grad_y_np, affine, header),
            f"{output_prefix}_tar_grad_y.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(tar_grad_z_np, affine, header),
            f"{output_prefix}_tar_grad_z.nii.gz",
        )

        # Save gradient magnitude images
        src_grad_mag = np.sqrt(
            src_grad_norm.view(b, c, *spatial_shape)[0, 0].cpu().numpy()
        )
        tar_grad_mag = np.sqrt(
            tar_grad_norm.view(b, c, *spatial_shape)[0, 0].cpu().numpy()
        )
        nib.save(
            nib.Nifti1Image(src_grad_mag, affine, header),
            f"{output_prefix}_src_grad_mag.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(tar_grad_mag, affine, header),
            f"{output_prefix}_tar_grad_mag.nii.gz",
        )

        # Save NGF field
        nib.save(
            nib.Nifti1Image(ngf_field_np, affine, header),
            f"{output_prefix}_ngf_field.nii.gz",
        )

        print(f"Saved gradient fields to {output_prefix}_*.nii.gz")

        # Return NGF metric for convenience
        ngf_metric = NormalizedGradientField3d(
            grad_method="default",
            gauss_sigma=gauss_sigma,
            eps=1e-5,
            mm_spacing=mm_spacing,
            reduction="mean",
        )
        ngf_value = -ngf_metric(src_data, tgt_data).item()
        return ngf_value


def main():
    """Run benchmark comparison between LaMAR and direct ANTs registration."""
    parser = argparse.ArgumentParser(
        description="Benchmark LaMAR vs direct ANTs registration"
    )
    parser.add_argument(
        "--moving", required=True, help="Input moving image to be registered"
    )
    parser.add_argument(
        "--fixed", required=True, help="Reference fixed image (target space)"
    )
    parser.add_argument(
        "--output-dir", default="benchmark_results", help="Directory for output files"
    )
    parser.add_argument(
        "--registration-method", default="SyNRA", help="Registration method"
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="Number of threads to use"
    )
    parser.add_argument(
        "--keep-files", action="store_true", help="Keep intermediate and output files"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress registration output"
    )
    parser.add_argument(
        "--quality-metric",
        default="mind",
        choices=["mind", "ngf", "nmi"],
        help="Metric for quality comparison (default: mind)",
    )
    parser.add_argument(
        "--save-gradients",
        action="store_true",
        help="Save normalized gradient fields as NIfTI files",
    )

    args = parser.parse_args()

    # Validate input files
    if not os.path.isfile(args.moving):
        print(f"Error: Moving image not found: {args.moving}")
        return 1
    if not os.path.isfile(args.fixed):
        print(f"Error: Fixed image not found: {args.fixed}")
        return 1

    # Create output directory or use temporary directory
    temp_dir = None
    if args.output_dir == "./benchmark_results":
        temp_dir = tempfile.mkdtemp(prefix="lamar_benchmark_")
        output_dir = temp_dir
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"Running benchmark with {args.threads} thread(s)...")
        print(f"Registration method: {args.registration_method}")

        # Run LaMAR registration
        print("\n--- Running LaMAR registration ---")
        lamar_time, lamar_output = run_lamar_registration(
            args.moving,
            args.fixed,
            output_dir,
            args.registration_method,
            args.threads,
            not args.quiet,
        )
        print(f"LaMAR registration completed in {lamar_time:.2f} seconds")

        # Run direct ANTs registration
        print("\n--- Running direct ANTs registration ---")
        ants_time, ants_output = run_direct_ants_registration(
            args.moving,
            args.fixed,
            output_dir,
            args.registration_method,
            args.threads,
            not args.quiet,
        )
        print(f"Direct ANTs registration completed in {ants_time:.2f} seconds")

        # Compare speeds
        speedup = ants_time / lamar_time if lamar_time > 0 else 0
        if speedup > 1:
            print(f"\nLaMAR is {speedup:.2f}x faster than direct ANTs registration")
        else:
            print(f"\nDirect ANTs is {1/speedup:.2f}x faster than LaMAR")

        # Compare quality
        print("\n--- Comparing registration quality ---")
        try:
            quality_results = compare_registration_quality(
                lamar_output, ants_output, args.fixed
            )

            # Display results for each metric
            for metric_name, scores in quality_results.items():
                if scores:
                    print(f"\n{metric_name.upper()} Metric:")
                    print(f"  LaMAR: {scores['lamar']:.4f}")
                    print(f"  ANTs:  {scores['ants']:.4f}")

                    diff = abs(scores["lamar"] - scores["ants"])
                    print(f"  Difference: {diff:.4f}")

                    if scores["lamar"] > scores["ants"]:
                        print("  LaMAR registration quality is higher")
                    else:
                        print("  Direct ANTs registration quality is higher")
        except Exception as e:
            print(f"Error comparing registration quality: {e}")

        # Save gradient fields if requested
        if args.save_gradients:
            print("\n--- Saving gradient fields ---")
            try:
                # Create a directory for the gradient fields
                gradient_fields_dir = os.path.join(output_dir, "gradient_fields")
                os.makedirs(gradient_fields_dir, exist_ok=True)

                # Save gradient fields for LaMAR registration
                print("Saving LaMAR gradient fields...")
                lamar_ngf = save_ngf_fields(
                    lamar_output,
                    args.fixed,
                    os.path.join(gradient_fields_dir, "lamar"),
                    gauss_sigma=0.5,
                )

                # Save gradient fields for ANTs registration
                print("Saving ANTs gradient fields...")
                ants_ngf = save_ngf_fields(
                    ants_output,
                    args.fixed,
                    os.path.join(gradient_fields_dir, "ants"),
                    gauss_sigma=0.5,
                )

                print(f"LaMAR NGF similarity: {lamar_ngf:.4f}")
                print(f"ANTs NGF similarity: {ants_ngf:.4f}")

            except Exception as e:
                print(f"Error saving gradient fields: {e}")

        print("\nBenchmark completed successfully!")

    except Exception as e:
        print(f"Error during benchmark: {e}")
        return 1
    finally:
        # Clean up temporary directory if created
        if temp_dir and not args.keep_files:
            shutil.rmtree(temp_dir)
        elif args.keep_files:
            print(f"\nOutput files kept in: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
