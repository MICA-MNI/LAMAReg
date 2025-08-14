#!/usr/bin/env python3
"""
Register a DWI b0 image to a T1-weighted image using FSL's epi_reg (BBR).

Usage:
    python run_epi_reg.py --b0 b0.nii.gz --t1 T1.nii.gz --t1brain T1_brain.nii.gz --out b0_to_T1

Requirements:
    - FSL installed and sourced (FSLDIR)
    - fslpy (pip install fslpy)

Author: Your Name
"""

import argparse
import os
from fsl.wrappers import epi_reg, flirt, applywarp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b0', required=True, help='Preprocessed b=0 image (NIfTI)')
    parser.add_argument('--t1', required=True, help='T1-weighted anatomical image (NIfTI)')
    parser.add_argument('--t1brain', required=True, help='Brain-extracted T1 image (NIfTI)')
    parser.add_argument('--out', required=True, help='Prefix for output files')
    parser.add_argument('--apply-nonlinear', action='store_true', 
                      help='Apply nonlinear warp in addition to affine transform')
    args = parser.parse_args()

    print(f"Running epi_reg with:\n  B0: {args.b0}\n  T1: {args.t1}\n  T1 Brain: {args.t1brain}")

    # Run epi_reg to generate the transformation matrix
    result = epi_reg(
        epi=args.b0,
        t1=args.t1,
        t1brain=args.t1brain,
        out=args.out
    )

    # Define output file paths
    affine_mat = f"{args.out}.mat"
    warp_file = f"{args.out}_warp.nii.gz"
    output_b0_in_t1 = f"{args.out}_in_T1space.nii.gz"
    
    print(f"Applying transformation to warp b0 to T1 space...")
    
    # Check if we should apply nonlinear warp (if it exists)
    if args.apply_nonlinear and os.path.exists(warp_file):
        print(f"Applying combined affine and nonlinear warp...")
        # Apply both affine and nonlinear transformations
        applywarp(
            in_file=args.b0,
            out=output_b0_in_t1,
            ref=args.t1,
            warp=warp_file,
            premat=affine_mat
        )
    else:
        # Apply just the affine transformation
        print(f"Applying affine transformation...")
        flirt(
            args.b0,
            args.t1,
            out=output_b0_in_t1,
            init=affine_mat,
            applyxfm=True
        )

    print("\nRegistration completed ✓")
    print(f"Transformation matrix saved: {affine_mat}")
    print(f"Transformed b0 image saved: {output_b0_in_t1}")
    
    # If there's also a warp file (nonlinear component)
    if os.path.exists(warp_file):
        print(f"Nonlinear warp field saved: {warp_file}")
        if not args.apply_nonlinear:
            print("Note: Nonlinear warp was not applied. Use --apply-nonlinear to include it.")


if __name__ == "__main__":
    main()
