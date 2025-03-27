#!/usr/bin/env python3
"""
Example script for contrast-agnostic registration using SynthSeg
"""

import os
import argparse
import subprocess
import sys


def synthseg_registration(input_image, reference_image, output_image, output_dir=None, input_parc=None, reference_parc=None, output_parc=None, generate_warpfield=False, apply_warpfield=False, registration_method="SyNRA"):
    """
    Perform contrast-agnostic registration using SynthSeg parcellation.
    """
    # Create output directory if specified
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    # Define paths for intermediate files
    if input_parc is None:
        input_parc = os.path.join(output_dir, "input_parcellation.nii.gz")
    if reference_parc is None:
        reference_parc = os.path.join(output_dir, "reference_parcellation.nii.gz")
    affine_transform = os.path.join(output_dir, "affine_transform.mat")
    warp_field = os.path.join(output_dir, "warp_field.nii.gz")
    inverse_warp = os.path.join(output_dir, "inverse_warp_field.nii.gz")
    
    print(f"Processing input image: {input_image}")
    print(f"Reference image: {reference_image}")
    print(f"Intermediate files will be saved in: {output_dir}")
    
    try:
        # Step 1: Generate parcellations with SynthSeg
        print("\n--- Step 1: Generating brain parcellations with SynthSeg ---")
        subprocess.run([
            "synthseg",  # Use entry point name
            "--i", input_image,
            "--o", input_parc,
            "--parc",
            "--cpu",
            "--threads", "1"
        ], check=True)
        
        subprocess.run([
            "synthseg",  # Use entry point name 
            "--i", reference_image,
            "--o", reference_parc,
            "--parc",
            "--cpu",
            "--threads", "1"
        ], check=True)
        
        # Step 2: Register parcellations using coregister
        print("\n--- Step 2: Coregistering parcellated images ---")
        subprocess.run([
            "coregister",  # Use entry point name
            "--fixed-file", reference_parc,
            "--moving-file", input_parc,
            "--output", os.path.join(output_dir, "registered_parcellation.nii.gz"),
            "--affine-file", affine_transform,
            "--warp-file", warp_field,
            "--rev-warp-file", inverse_warp,
            "--rev-affine-file", os.path.join(output_dir, "inverse_affine_transform.mat")
        ], check=True)
        
        # Step 3: Apply transformation to the original input image
        print("\n--- Step 3: Applying transformation to original input image ---")
        subprocess.run([
            "apply_warp",  # Use entry point name
            "--moving", input_image,
            "--reference", reference_image,
            "--affine", affine_transform,
            "--warp", warp_field,
            "--output", output_image
        ], check=True)
        
        print(f"\nSuccess! Registered image saved to: {output_image}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Contrast-agnostic registration using SynthSeg")
    parser.add_argument("--moving", required=True, help="Input moving image to be registered")
    parser.add_argument("--fixed", required=True, help="Reference fixed image (target space)")
    parser.add_argument("--output", required=True, help="Output registered image")
    parser.add_argument("--workdir", help="Directory for intermediate files (default: current directory)")
    parser.add_argument("--moving-parc", help="Input moving parcellation")
    parser.add_argument("--fixed-parc", help="Reference fixed parcellation")
    parser.add_argument("--output-parc", help="Output registered parcellation")
    parser.add_argument("--generate-warpfield", action="store_true", help="Generate warp field for registration")
    parser.add_argument("--apply-warpfield", action="store_true", help="Apply warp field to moving image")
    parser.add_argument("--registration-method", default="SyNRA", help="Registration method")
    args = parser.parse_args()
    
    synthseg_registration(
        input_image=args.moving,
        reference_image=args.fixed,
        output_image=args.output,
        output_dir=args.workdir,
        input_parc=args.moving_parc,
        reference_parc=args.fixed_parc,
        output_parc=args.output_parc,
        generate_warpfield=args.generate_warpfield,
        apply_warpfield=args.apply_warpfield,
        registration_method=args.registration_method
    )


if __name__ == "__main__":
    main()