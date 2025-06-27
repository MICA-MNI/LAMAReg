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
import ants

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


def run_lamar_registration_robust(
    moving_img,
    fixed_img,
    output_dir,
    registration_method="SyNRA",
    threads=1,
    verbose=True,
):
    """Run registration using LaMAR coregister with initial transform and measure time."""
    start_time = time.time()

    # Set up output paths
    output_img = os.path.join(output_dir, "lamar_robust_registered.nii.gz")
    registered_parc = os.path.join(output_dir, "registered_parc_robust.nii.gz")
    affine_file = os.path.join(output_dir, "lamar_robust_affine.mat")
    warp_file = os.path.join(output_dir, "lamar_robust_warp.nii.gz")
    
    # Initial transform files from standard registration
    initial_affine_file = os.path.join(output_dir, "lamar_affine.mat")
    initial_warp_file = os.path.join(output_dir, "lamar_warp.nii.gz")

    # Build command for lamar coregister with initial transforms
    cmd = [
        "lamar", "coregister",
        "--fixed-file", fixed_img,
        "--moving-file", moving_img,
        "--output", output_img,
        "--warp-file", warp_file,
        "--affine-file", affine_file,
        "--registration-method", "SyNOnly",
        "--initial-affine-file", initial_affine_file,
        "--initial-warp-file", initial_warp_file,
        "--interpolator", "linear",
        "--reg-iterations","10, 20",
    ]
    
    # Add additional parameters
    if verbose:
        cmd.append("--verbose")
    
    # Add thread parameters if coregister supports them
    if threads > 1:
        env = os.environ.copy()
        env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
        env["OMP_NUM_THREADS"] = str(threads)

    # Run LaMAR coregister
    if verbose:
        subprocess.run(cmd, check=True, env=env if threads > 1 else None)
    else:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                      env=env if threads > 1 else None)

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
        reg_iterations=(10,20)
    )

    # Save outputs
    ants.image_write(registration["warpedmovout"], output_img)

    temp_files_to_delete = set(registration['fwdtransforms'] + registration['invtransforms'])
    deleted_count = 0
    for temp_file in temp_files_to_delete:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                deleted_count += 1
        except OSError as e:
            print(f"Warning: Could not remove temporary file {temp_file}: {e}")
    print(f"Successfully cleaned up {deleted_count} temporary files.")

    elapsed_time = time.time() - start_time
    return elapsed_time, output_img


def run_direct_ants_registration_default(
    moving_img,
    fixed_img,
    output_dir,
    registration_method="SyNRA",
    threads=1,
    verbose=True,
):
    """Run direct ANTs registration via ANTsPyX with default parameters and measure time."""
    import ants

    start_time = time.time()

    # Set up output paths
    output_img = os.path.join(output_dir, "direct_ants_default_registered.nii.gz")

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
        print(f"Running ANTsPyX registration with default parameters, method: {type_of_transform}")
        print(f"Thread count: {threads}")
        print(f"Moving image: {moving_img}")
        print(f"Fixed image: {fixed_img}")

    # Perform registration with default parameters
    registration = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform=type_of_transform,
        verbose=verbose,
        # No reg_iterations parameter means using ANTs defaults
    )

    # Save outputs
    ants.image_write(registration["warpedmovout"], output_img)

    temp_files_to_delete = set(registration['fwdtransforms'] + registration['invtransforms'])
    deleted_count = 0
    for temp_file in temp_files_to_delete:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                deleted_count += 1
        except OSError as e:
            print(f"Warning: Could not remove temporary file {temp_file}: {e}")
    print(f"Successfully cleaned up {deleted_count} temporary files.")

    elapsed_time = time.time() - start_time
    return elapsed_time, output_img


def compare_registration_quality(lamar_output, ants_output, fixed_img, lamar_robust_output=None, ants_default_output=None):
    """Compare the registration quality using all available metrics.

    Args:
        lamar_output: Path to LaMAR registered image
        ants_output: Path to ANTs registered image
        fixed_img: Path to fixed reference image
        lamar_robust_output: Path to LaMAR registered image with robust flag (optional)
        ants_default_output: Path to ANTs registered image with default parameters (optional)

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

    # Load robust LAMAReg and ANTs default if provided
    lamar_robust_data = None
    lamar_robust_tensor = None
    ants_default_data = None
    ants_default_tensor = None
    
    if lamar_robust_output is not None:
        lamar_robust_nib = nib.load(lamar_robust_output)
        lamar_robust_data = lamar_robust_nib.get_fdata()
        lamar_robust_tensor = torch.from_numpy(lamar_robust_data).float().unsqueeze(0).unsqueeze(0)
        
    if ants_default_output is not None:
        ants_default_nib = nib.load(ants_default_output)
        ants_default_data = ants_default_nib.get_fdata()
        ants_default_tensor = torch.from_numpy(ants_default_data).float().unsqueeze(0).unsqueeze(0)

    # Convert to PyTorch tensors
    lamar_tensor = torch.from_numpy(lamar_img_data).float()
    ants_tensor = torch.from_numpy(ants_img_data).float()
    fixed_tensor = torch.from_numpy(fixed_img_data).float()

    # Add batch and channel dimensions
    lamar_tensor = lamar_tensor.unsqueeze(0).unsqueeze(0)
    ants_tensor = ants_tensor.unsqueeze(0).unsqueeze(0)
    fixed_tensor = fixed_tensor.unsqueeze(0).unsqueeze(0)

    results = {}

    # Calculate Mutual Information using ANTsPy
    def mutual_information(img1, img2, bins=32):
        """Calculate mutual information between two images using ANTsPy."""
        # Convert numpy arrays to ANTs images
        img1_ants = ants.from_numpy(img1.astype(np.float32))
        img2_ants = ants.from_numpy(img2.astype(np.float32))
        
        # Calculate mutual information directly with ANTsPy
        return ants.image_mutual_information(img1_ants, img2_ants)

    # Calculate ANTSNeighborhoodCorrelation using ANTsPy
    def ants_neighborhood_correlation(img1, img2):
        """Calculate ANTSNeighborhoodCorrelation between two images using ANTsPy."""
        # Convert numpy arrays to ANTs images
        img1_ants = ants.from_numpy(img1.astype(np.float32))
        img2_ants = ants.from_numpy(img2.astype(np.float32))
        
        # Use ANTSNeighborhoodCorrelation metric directly
        similarity = ants.image_similarity(img1_ants, img2_ants, metric_type='ANTSNeighborhoodCorrelation')
        
        return similarity

    # Calculate and store MI
    results["mi"] = {
        "lamar": mutual_information(lamar_img_data, fixed_img_data),
        "ants": mutual_information(ants_img_data, fixed_img_data),
    }
    
    # Calculate and store ANTSNeighborhoodCorrelation
    results["antsneighborhoodcorrelation"] = {
        "lamar": ants_neighborhood_correlation(lamar_img_data, fixed_img_data),
        "ants": ants_neighborhood_correlation(ants_img_data, fixed_img_data),
    }
    
    # Add robust and default results if available
    if lamar_robust_data is not None:
        results["mi"]["lamar_robust"] = mutual_information(lamar_robust_data, fixed_img_data)
        results["antsneighborhoodcorrelation"]["lamar_robust"] = ants_neighborhood_correlation(lamar_robust_data, fixed_img_data)
        
    if ants_default_data is not None:
        results["mi"]["ants_default"] = mutual_information(ants_default_data, fixed_img_data)
        results["antsneighborhoodcorrelation"]["ants_default"] = ants_neighborhood_correlation(ants_default_data, fixed_img_data)

    # Try MIND metric
    try:
        from torch_mind import MINDLoss3D

        mind_loss = MINDLoss3D()

        with torch.no_grad():
            lamar_mind = mind_loss(lamar_tensor, fixed_tensor).item()
            ants_mind = mind_loss(ants_tensor, fixed_tensor).item()
            
            mind_results = {"lamar": lamar_mind, "ants": ants_mind}
            
            if lamar_robust_tensor is not None:
                mind_results["lamar_robust"] = mind_loss(lamar_robust_tensor, fixed_tensor).item()
            
            if ants_default_tensor is not None:
                mind_results["ants_default"] = mind_loss(ants_default_tensor, fixed_tensor).item()
            
            results["mind"] = mind_results
    except Exception as e:
        print(f"Error calculating MIND: {e}")
        results["mind"] = None

    # Try NGF metric
    try:
        from normalized_gradient_field import NormalizedGradientField3d

        pixel_spacing = lamar_img_nib.header.get_zooms()[:3]

        ngf = NormalizedGradientField3d(
            grad_method="default",
            mm_spacing=pixel_spacing,
            reduction="mean",
        )

        with torch.no_grad():
            lamar_ngf = ngf(lamar_tensor, fixed_tensor).item()
            ants_ngf = ngf(ants_tensor, fixed_tensor).item()
            
            ngf_results = {"lamar": lamar_ngf, "ants": ants_ngf}
            
            if lamar_robust_tensor is not None:
                ngf_results["lamar_robust"] = ngf(lamar_robust_tensor, fixed_tensor).item()
                
            if ants_default_tensor is not None:
                ngf_results["ants_default"] = ngf(ants_default_tensor, fixed_tensor).item()
            
            results["ngf"] = ngf_results
    except Exception as e:
        print(f"Error calculating NGF: {e}")
        results["ngf"] = None

    return results


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
