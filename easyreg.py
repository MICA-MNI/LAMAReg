#!/usr/bin/env python3
"""
Register images using FreeSurfer's mri_easyreg tool.

This script provides a wrapper for FreeSurfer's mri_easyreg, which performs
symmetric diffeomorphic registration between two images using deep learning.

Usage:
    python easyreg.py --ref ref.nii.gz --flo moving.nii.gz --out output_prefix

Requirements:
    - FreeSurfer installed and sourced (FREESURFER_HOME)
    - FreeSurfer's mri_easyreg tool available on PATH

Author: LAMAR Team
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import nibabel as nib

def run_command(cmd, verbose=False):
    """Run a shell command and handle errors."""
    if verbose:
        print(f"Running command: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error executing command: {' '.join(cmd)}")
            print(f"STDERR: {stderr}")
            return False
        if verbose:
            print(stdout)
        return True
    except Exception as e:
        print(f"Exception running command: {e}")
        return False


def check_freesurfer():
    """Check if FreeSurfer is properly installed and sourced."""
    freesurfer_home = os.environ.get("FREESURFER_HOME")
    if not freesurfer_home:
        print("ERROR: FREESURFER_HOME environment variable not set.")
        print("Please make sure FreeSurfer is installed and sourced.")
        return False
    
    # Check if mri_easyreg exists
    try:
        subprocess.run(["which", "mri_easyreg"], 
                      check=True, 
                      stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        print("ERROR: mri_easyreg not found in PATH.")
        print("Please make sure FreeSurfer is properly installed and sourced.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Register images using FreeSurfer's mri_easyreg")
    
    # Required arguments
    parser.add_argument('--ref', required=True, help='Reference image (.nii.gz or .mgz)')
    parser.add_argument('--flo', required=True, help='Floating/moving image (.nii.gz or .mgz)')
    parser.add_argument('--out', required=True, help='Prefix for output files')
    
    # Optional arguments
    parser.add_argument('--ref_seg', help='Reference image segmentation (will be generated if not provided)')
    parser.add_argument('--flo_seg', help='Floating image segmentation (will be generated if not provided)')
    parser.add_argument('--threads', type=int, default=1, 
                      help='Number of threads to use (default: 1, set -1 for all cores)')
    parser.add_argument('--affine_only', action='store_true',
                      help='Perform only affine registration (skip nonlinear)')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    
    args = parser.parse_args()

    # Check if FreeSurfer is available
    if not check_freesurfer():
        sys.exit(1)

    # Create output directory if it doesn't exist
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Prepare output file paths
    ref_seg_path = args.ref_seg if args.ref_seg else f"{args.out}_ref_seg.nii.gz"
    flo_seg_path = args.flo_seg if args.flo_seg else f"{args.out}_flo_seg.nii.gz"
    ref_reg_path = f"{args.out}_ref_in_flo.nii.gz"
    flo_reg_path = f"{args.out}_flo_in_ref.nii.gz"
    fwd_field_path = f"{args.out}_fwd_field.nii.gz"
    bak_field_path = f"{args.out}_bak_field.nii.gz"
    
    # Build the mri_easyreg command
    cmd = [
        "mri_easyreg",
        "--ref", args.ref,
        "--flo", args.flo,
        "--ref_seg", ref_seg_path,
        "--flo_seg", flo_seg_path,
        "--ref_reg", ref_reg_path,
        "--flo_reg", flo_reg_path,
        "--fwd_field", fwd_field_path,
        "--bak_field", bak_field_path,
        "--threads", str(args.threads),
    ]
    
  
    
    # Print registration information
    print(f"Running mri_easyreg with:")
    print(f"  Reference image: {args.ref}")
    print(f"  Floating image: {args.flo}")
    print(f"  Reference segmentation: {ref_seg_path}" + 
          (" (will be generated)" if not args.ref_seg else ""))
    print(f"  Floating segmentation: {flo_seg_path}" + 
          (" (will be generated)" if not args.flo_seg else ""))
    print(f"  Registration mode: {'Affine only' if args.affine_only else 'Affine + Nonlinear'}")
    print(f"  Threads: {args.threads}")
    
    # Run registration
    if not run_command(cmd, args.verbose):
        print("ERROR: Registration failed")
        sys.exit(1)
    
    # Check if output files exist
    for file_path, file_desc in [
        (flo_reg_path, "Deformed floating image"),
        (ref_reg_path, "Deformed reference image"),
        (fwd_field_path, "Forward deformation field"),
        (bak_field_path, "Backward deformation field")
    ]:
        if not os.path.exists(file_path):
            print(f"WARNING: {file_desc} {file_path} not created")
    
    print("\nRegistration completed ✓")
    print(f"Segmentations saved:")
    print(f"  Reference: {ref_seg_path}")
    print(f"  Floating: {flo_seg_path}")
    
    print(f"Deformed images saved:")
    print(f"  Reference in floating space: {ref_reg_path}")
    print(f"  Floating in reference space: {flo_reg_path}")
    
    print(f"Deformation fields saved:")
    print(f"  Forward field: {fwd_field_path}")
    print(f"  Backward field: {bak_field_path}")

    # Squeeze output files to save space
    output = nib.load(flo_reg_path)
    output_squeezed = nib.Nifti1Image(output.get_fdata().squeeze(), output.affine, output.header)
    output_squeezed.to_filename(flo_reg_path)


if __name__ == "__main__":
    main()