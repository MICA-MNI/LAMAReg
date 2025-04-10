#!/usr/bin/env python3
"""
Full pipeline test for LaMAR.

This script runs a complete registration workflow using the example data
and verifies that all output files are created correctly.
"""

import os
import sys
import subprocess
import nibabel as nib
import shutil
import time
import argparse


def test_full_pipeline(quick=False):
    """Run LaMAR pipeline on example data and check outputs."""
    print("\n=== LaMAR Full Pipeline Test ===")

    # Define paths
    example_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "example_data"
    )
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_output")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    moving_file = os.path.join(example_dir, "sub-HC001_ses-02_space-dwi_desc-b0.nii.gz")

    fixed_file = os.path.join(example_dir, "sub-HC001_ses-01_T1w.nii.gz")

    # Check if example data exists
    if not os.path.exists(moving_file):
        print(f"ERROR: Moving image not found: {moving_file}")
        return False
    if not os.path.exists(fixed_file):
        print(f"ERROR: Fixed image not found: {fixed_file}")
        return False

    # Define output files
    output_image = os.path.join(output_dir, "sub-001_dwi_in_T1w.nii.gz")
    moving_parc = os.path.join(output_dir, "sub-001_dwi_parc.nii.gz")
    fixed_parc = os.path.join(output_dir, "sub-001_T1w_parc.nii.gz")
    registered_parc = os.path.join(output_dir, "sub-001_dwi_reg_parc.nii.gz")
    affine_file = os.path.join(output_dir, "dwi_to_T1w_affine.mat")
    warp_file = os.path.join(output_dir, "dwi_to_T1w_warp.nii.gz")
    inverse_warp = os.path.join(output_dir, "T1w_to_dwi_warp.nii.gz")
    inverse_affine = os.path.join(output_dir, "T1w_to_dwi_affine.mat")
    qc_csv = os.path.join(output_dir, "dice_scores.csv")

    # Clean previous outputs
    for f in [
        output_image,
        moving_parc,
        fixed_parc,
        registered_parc,
        affine_file,
        warp_file,
        inverse_warp,
        inverse_affine,
        qc_csv,
    ]:
        if os.path.exists(f):
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)

    # Build the LaMAR command
    cmd = [
        "lamar",
        "register",
        "--moving",
        moving_file,
        "--fixed",
        fixed_file,
        "--output",
        output_image,
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
        "--inverse-warpfield",
        inverse_warp,
        "--inverse-affine",
        inverse_affine,
        "--qc-csv",
        qc_csv,
        "--synthseg-threads",
        "2",  # Lower thread count for CI
        "--ants-threads",
        "2",  # Lower thread count for CI
    ]

    print("\nRunning LaMAR pipeline...")
    print(" ".join(cmd))
    start_time = time.time()

    try:
        # Run the pipeline
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: Command failed with return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Check if expected output files exist
        expected_files = [
            output_image,
            moving_parc,
            fixed_parc,
            registered_parc,
            affine_file,
            warp_file,
            inverse_warp,
            inverse_affine,
        ]

        missing_files = []
        for f in expected_files:
            if not os.path.exists(f):
                missing_files.append(f)

        if missing_files:
            print("ERROR: The following output files were not created:")
            for f in missing_files:
                print(f"  - {f}")
            return False

        # Verify outputs
        try:
            # Check that images can be loaded
            nib.load(output_image)
            nib.load(moving_parc)
            nib.load(fixed_parc)
            nib.load(registered_parc)

            # Check that transform files are not empty
            if os.path.getsize(affine_file) == 0:
                print(f"ERROR: Affine file is empty: {affine_file}")
                return False
            if os.path.getsize(warp_file) == 0:
                print(f"ERROR: Warp file is empty: {warp_file}")
                return False

            # Check Dice scores if available
            if os.path.exists(qc_csv):
                print("\nRegistration quality metrics:")
                with open(qc_csv, "r") as f:
                    print(f.read())

        except Exception as e:
            print(f"ERROR: Failed to verify output files: {e}")
            return False

        elapsed_time = time.time() - start_time
        print(f"\nTest completed successfully in {elapsed_time:.1f} seconds!")
        return True

    except Exception as e:
        print(f"ERROR: Test failed with exception: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a full pipeline test for LaMAR")
    args = parser.parse_args()

    success = test_full_pipeline()
    sys.exit(0 if success else 1)
