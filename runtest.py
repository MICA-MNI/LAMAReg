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
    """Compare the registration quality between the two methods using normalized mutual information."""
    # Load images
    lamar_img = nib.load(lamar_output).get_fdata()
    ants_img = nib.load(ants_output).get_fdata()
    fixed_img_data = nib.load(fixed_img).get_fdata()

    # Calculate normalized mutual information
    def normalized_mutual_information(img1, img2, bins=32):
        """
        Calculate normalized mutual information between two images.
        Higher values indicate better alignment between images.
        """
        # Flatten arrays
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()

        # Calculate histograms
        hist_joint, _, _ = np.histogram2d(img1_flat, img2_flat, bins=bins)
        hist_img1 = np.sum(hist_joint, axis=1)
        hist_img2 = np.sum(hist_joint, axis=0)

        # Normalize histograms to get probability distributions
        p_img1 = hist_img1 / np.sum(hist_img1)
        p_img2 = hist_img2 / np.sum(hist_img2)
        p_joint = hist_joint / np.sum(hist_joint)

        # Calculate entropies - handle zero probabilities
        eps = np.finfo(float).eps
        h_img1 = -np.sum(p_img1 * np.log2(p_img1 + eps))
        h_img2 = -np.sum(p_img2 * np.log2(p_img2 + eps))
        h_joint = -np.sum(p_joint * np.log2(p_joint + eps))

        # Calculate normalized mutual information
        return (h_img1 + h_img2) / h_joint if h_joint > 0 else 0

    # Calculate NMI scores
    lamar_nmi = normalized_mutual_information(lamar_img, fixed_img_data)
    ants_nmi = normalized_mutual_information(ants_img, fixed_img_data)

    return lamar_nmi, ants_nmi


def main():
    """Run benchmark comparison between LaMAR and direct ANTs registration."""
    parser = argparse.ArgumentParser(
        description="Compare registration speed between LaMAR and direct ANTs"
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
            lamar_nmi, ants_nmi = compare_registration_quality(
                lamar_output, ants_output, args.fixed
            )
            print(
                f"LaMAR normalized mutual information with fixed image: {lamar_nmi:.4f}"
            )
            print(
                f"Direct ANTs normalized mutual information with fixed image: {ants_nmi:.4f}"
            )

            print(f"\nQuality difference: {abs(lamar_nmi - ants_nmi):.4f}")
            if lamar_nmi > ants_nmi:
                print("LaMAR registration quality is higher")
            else:
                print("Direct ANTs registration quality is higher")
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
