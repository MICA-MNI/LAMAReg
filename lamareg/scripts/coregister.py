"""
coregister - Image Registration for Aligning Neuroimaging Data

Part of the micaflow processing pipeline for neuroimaging data.

This module performs comprehensive image registration between two images using the
Advanced Normalization Tools (ANTs) SyNRA algorithm, which combines rigid, affine,
and symmetric normalization transformations. It aligns a moving image with a fixed
reference space, enabling spatial normalization of neuroimaging data for group analysis,
multimodal integration, or atlas-based analyses.

Features:
--------
- Combined rigid, affine, and SyN nonlinear registration in one step
- Bidirectional transformation capability (forward and inverse)
- Option to save all transformation components for later application
- Uses ANTs' powerful SyNRA algorithm for optimal accuracy
- Preserves header information in the registered output images

API Usage:
---------
micaflow coregister
    --fixed-file <path/to/reference.nii.gz>
    --moving-file <path/to/source.nii.gz>
    --output <path/to/registered.nii.gz>
    [--warp-file <path/to/warp.nii.gz>]
    [--affine-file <path/to/affine.mat>]
    [--rev-warp-file <path/to/reverse_warp.nii.gz>]
    [--rev-affine-file <path/to/reverse_affine.mat>]

Python Usage:
-----------
>>> from micaflow.scripts.coregister import ants_linear_nonlinear_registration
>>> ants_linear_nonlinear_registration(
...     fixed_file="mni152.nii.gz",
...     moving_file="subject_t1w.nii.gz",
...     out_file="registered_t1w.nii.gz",
...     warp_file="warp.nii.gz",
...     affine_file="affine.mat",
...     rev_warp_file="reverse_warp.nii.gz",
...     rev_affine_file="reverse_affine.mat"
... )

"""

import ants
import argparse
import shutil
import sys
from colorama import init, Fore, Style
import os
import multiprocessing
import nibabel as nib
import numpy as np
import tempfile


init()

# Get number of available CPU cores
DEFAULT_THREADS = multiprocessing.cpu_count()


def print_help():
    """Print a help message with examples."""
    # ANSI color codes
    CYAN = Fore.CYAN
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    BLUE = Fore.BLUE
    MAGENTA = Fore.MAGENTA
    BOLD = Style.BRIGHT
    RESET = Style.RESET_ALL

    help_text = f"""
    {CYAN}{BOLD}╔════════════════════════════════════════════════════════════════╗
    ║                      IMAGE COREGISTRATION                      ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}

    This script performs linear (rigid + affine) and nonlinear (SyN) registration 
    between two images using ANTs. The registration aligns the moving image to 
    match the fixed reference image space.

    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow coregister {GREEN}[options]{RESET}

    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--fixed{RESET}   : Path to the fixed/reference image (.nii.gz)
      {YELLOW}--moving{RESET}  : Path to the moving image to be registered (.nii.gz)

    {CYAN}{BOLD}─────────────────── OPTIONAL ARGUMENTS ───────────────────{RESET}
      {YELLOW}--output{RESET}             : Output path for the registered image (.nii.gz)
      {YELLOW}--registration-method{RESET}: Registration method (default: "SyNRA")
      {YELLOW}--warp-file{RESET}          : Path to save the forward warp field (.nii.gz)
      {YELLOW}--affine-file{RESET}        : Path to save the forward affine transform (.mat)
      {YELLOW}--inverse-warp-file{RESET}  : Path to save the reverse warp field (.nii.gz)
      {YELLOW}--inverse-affine-file{RESET}: Path to save the reverse affine transform (.mat)
      {YELLOW}--initial-affine-file{RESET}: Path to initial affine transform to use (.mat)
      {YELLOW}--initial-warp-file{RESET}  : Path to initial warp field to use (.nii.gz)
      {YELLOW}--interpolator{RESET}       : Interpolation method (default: "genericLabel")
      {YELLOW}--threads{RESET}            : Number of CPU threads (default: all available)

    {CYAN}{BOLD}───────────────── ANTS REGISTRATION OPTIONS ────────────────{RESET}
      {YELLOW}--verbose{RESET}             : Enable verbose output
      {YELLOW}--grad-step{RESET}           : Gradient step size (default: 0.2)
      {YELLOW}--flow-sigma{RESET}          : Smoothing for update field (default: 3)
      {YELLOW}--total-sigma{RESET}         : Smoothing for total field (default: 0)
      {YELLOW}--aff-metric{RESET}          : Metric for affine stage (default: "mattes")
      {YELLOW}--aff-sampling{RESET}        : Sampling parameter for affine metric (default: 32)
      {YELLOW}--syn-metric{RESET}          : Metric for SyN stage (default: "mattes")
      {YELLOW}--syn-sampling{RESET}        : Sampling parameter for SyN metric (default: 32)
      {YELLOW}--reg-iterations{RESET}      : SyN iterations, comma-separated (e.g., "40,20,0")
      {YELLOW}--aff-iterations{RESET}      : Affine iterations, comma-separated (e.g., "2100,1200,1200,10")
      {YELLOW}--aff-shrink-factors{RESET}  : Affine shrink factors, comma-separated (e.g., "6,4,2,1")
      {YELLOW}--aff-smoothing-sigmas{RESET}: Affine smoothing sigmas, comma-separated (e.g., "3,2,1,0")
      {YELLOW}--random-seed{RESET}         : Random seed for reproducibility

    {CYAN}{BOLD}────────────────── EXAMPLE USAGE ────────────────────────{RESET}

    {BLUE}# Basic registration with default parameters{RESET}
    micaflow coregister {YELLOW}--fixed{RESET} mni152.nii.gz {YELLOW}--moving{RESET} subject_t1w.nii.gz \\
      {YELLOW}--output{RESET} registered_t1w.nii.gz

    {BLUE}# Registration with saved transforms{RESET}
    micaflow coregister {YELLOW}--fixed{RESET} mni152.nii.gz {YELLOW}--moving{RESET} subject_t1w.nii.gz \\
      {YELLOW}--output{RESET} registered_t1w.nii.gz {YELLOW}--warp-file{RESET} warp.nii.gz {YELLOW}--affine-file{RESET} affine.mat

    {BLUE}# Registration with custom ANTs parameters{RESET}
    micaflow coregister {YELLOW}--fixed{RESET} mni152.nii.gz {YELLOW}--moving{RESET} subject_t1w.nii.gz \\
      {YELLOW}--output{RESET} registered_t1w.nii.gz {YELLOW}--reg-iterations{RESET} "100,50,20" \\
      {YELLOW}--grad-step{RESET} 0.1 {YELLOW}--verbose{RESET}

    {CYAN}{BOLD}────────────────────────── NOTES ───────────────────────{RESET}
    {MAGENTA}•{RESET} The registration performs SyNRA transformation (rigid+affine+SyN)
    {MAGENTA}•{RESET} Forward transforms convert from moving space to fixed space
    {MAGENTA}•{RESET} Reverse transforms convert from fixed space to moving space
    {MAGENTA}•{RESET} The transforms can be applied to other images using apply_warp
    {MAGENTA}•{RESET} For reproducible results, set the random seed
    """
    print(help_text)


def combine_warps_and_transform(
    first_warp_path, 
    second_warp_path,
    combined_warp_output_path
):
    """
    Combines two warp fields, resampling the first warp to match the second warp if needed.
    
    Args:
        first_warp_path (str): Path to the first warp field
        second_warp_path (str): Path to the second warp field
        combined_warp_output_path (str): Path to save the combined warp field
    """
    # Load warp fields
    first_warp_nib = nib.load(first_warp_path)
    second_warp_nib = nib.load(second_warp_path)
    first_arr = first_warp_nib.get_fdata().squeeze()
    second_arr = second_warp_nib.get_fdata().squeeze()

    # Add the displacements (this combines the transforms)
    combined_arr = first_arr + second_arr
    
    # Create and save the combined warp field
    combined_warp = nib.Nifti1Image(combined_arr, second_warp_nib.affine, second_warp_nib.header)
    combined_warp.to_filename(combined_warp_output_path)
    print(f"Combined warp field saved as {combined_warp_output_path}")


def ants_linear_nonlinear_registration(
    fixed_file,
    moving_file,
    out_file=None,
    warp_file=None,
    affine_file=None,
    rev_warp_file=None,
    rev_affine_file=None,
    registration_method="SyNRA",
    initial_affine_file=None,
    initial_warp_file=None,
    interpolator="genericLabel",
    threads=DEFAULT_THREADS,  # Use all available cores by default
    **kwargs,  # Add this to accept extra parameters
):
    """Perform linear (rigid + affine) and nonlinear registration using ANTsPy.

    This function performs registration between two images using ANTs' SyNRA transform,
    which includes both linear (rigid + affine) and nonlinear (SyN) components.

    Args:
        fixed_file (str): Path to the fixed/reference image.
        moving_file (str): Path to the moving image that will be registered.
        out_file (str, optional): Path where the registered image will be saved.
        warp_file (str, optional): Path to save the forward warp field.
        affine_file (str, optional): Path to save the forward affine transform.
        rev_warp_file (str, optional): Path to save the reverse warp field.
        rev_affine_file (str, optional): Path to save the reverse affine transform.
        registration_method (str): Registration method to use. Defaults to "SyNRA".
        initial_affine_file (str, optional): Path to initial affine transform.
        initial_warp_file (str, optional): Path to initial warp field.
        interpolator (str): Interpolation method. Defaults to "genericLabel".
        threads (int): Number of threads to use for registration. Defaults to all available cores.
        **kwargs: Additional arguments passed directly to ants.registration
                 Examples: verbose, grad_step, reg_iterations, etc.

    Returns:
        None: The function saves the registered image and transform files to disk.
    """
    if (
        not out_file
        and not warp_file
        and not affine_file
        and not rev_warp_file
        and not rev_affine_file
    ):
        print(Fore.RED + "Error: No outputs specified." + Style.RESET_ALL)
        sys.exit(1)

    # Set ANTs/ITK thread count in environment variables
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)

    # Load images
    fixed = ants.image_read(fixed_file)
    moving = ants.image_read(moving_file)
    initial_transform = []

    if initial_warp_file:
        initial_transform.append(initial_warp_file)

    if initial_affine_file:
        initial_transform.append(initial_affine_file)

    if initial_transform == []:
        initial_transform = None


    # Pass all arguments to ants.registration, including any extra kwargs
    transforms = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=registration_method,
        # interpolator=interpolator,
        initial_transform=initial_transform,
        **kwargs,  # Pass through all additional arguments
    )
    print("Applying transforms...")
    registered = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=transforms["fwdtransforms"],
        interpolator=interpolator,
    )
    print("Transforms applied successfully.")
    # Save the registered moving image
    if out_file is not None:
        ants.image_write(registered, out_file)
        print(f"Registration complete. Saved registered image as {out_file}")

    if warp_file:
        if initial_warp_file:
            combine_warps_and_transform(
                transforms["fwdtransforms"][0],
                transforms["fwdtransforms"][1],
                warp_file,)
            print(f"Saved warp field as {warp_file}")
        else:
            shutil.copyfile(transforms["fwdtransforms"][0], warp_file)
            print(f"Saved warp field as {warp_file}")

    if affine_file:
        if initial_affine_file:
            shutil.copyfile(transforms["fwdtransforms"][2], affine_file)
            print(f"Saved affine transform as {affine_file}")
        else:
            shutil.copyfile(transforms["fwdtransforms"][1], affine_file)
            print(f"Saved affine transform as {affine_file}")
    if rev_warp_file:
        if initial_warp_file:
            combine_warps_and_transform(
                transforms["invtransforms"][2],
                transforms["invtransforms"][1],
                warp_file,)
        else:
            shutil.copyfile(transforms["invtransforms"][1], rev_warp_file)
            print(f"Saved reverse warp field as {rev_warp_file}")
    if rev_affine_file:
        shutil.copyfile(transforms["invtransforms"][0], rev_affine_file)
        print(f"Saved reverse affine transform as {rev_affine_file}")

    print("All specified outputs saved successfully.")
    print("Cleaning up temporary files...")
    temp_files_to_delete = set(transforms['fwdtransforms'] + transforms['invtransforms'])
    deleted_count = 0
    for temp_file in temp_files_to_delete:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                deleted_count += 1
        except OSError as e:
            print(f"Warning: Could not remove temporary file {temp_file}: {e}")
    print(f"Successfully cleaned up {deleted_count} temporary files.")


def main():
    """Entry point for command-line use"""
    parser = argparse.ArgumentParser(description="Coregistration tool")

    # Required/standard arguments
    parser.add_argument("--fixed", required=True, help="Fixed/reference image file path")
    parser.add_argument("--moving", required=True, help="Moving image file path")
    parser.add_argument("--output", help="Output image file path")
    parser.add_argument(
        "--registration-method", default="SyNRA", help="Registration method"
    )
    parser.add_argument("--affine-file", help="Affine transformation file path")
    parser.add_argument("--warp-file", help="Warp field file path")
    parser.add_argument(
        "--inverse-warp-file", help="Inverse warp field file path"  # Standardized name
    )
    parser.add_argument(
        "--inverse-affine-file", help="Inverse affine transformation file path"  # Standardized name
    )
    parser.add_argument(
        "--initial-affine-file", help="Initial affine transformation file path"
    )
    parser.add_argument("--initial-warp-file", help="Initial warp field file path")
    parser.add_argument(
        "--interpolator", help="Interpolator type", default="genericLabel"
    )
    
    # Add threads parameter with default value of all cores
    parser.add_argument(
        "--threads", 
        type=int, 
        default=DEFAULT_THREADS, 
        help=f"Number of threads to use (default: {DEFAULT_THREADS} - all cores)"
    )

    # Add common ANTs registration parameters
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--grad-step", type=float, default=0.2, help="Gradient step size (default: 0.2)"
    )
    parser.add_argument(
        "--flow-sigma",
        type=float,
        default=3,
        help="Smoothing for update field (default: 3)",
    )
    parser.add_argument(
        "--total-sigma",
        type=float,
        default=0,
        help="Smoothing for total field (default: 0)",
    )
    parser.add_argument(
        "--aff-metric",
        default="mattes",
        help="Metric for affine stage (default: mattes)",
    )
    parser.add_argument(
        "--aff-sampling",
        type=int,
        default=32,
        help="Sampling parameter for affine metric (default: 32)",
    )
    parser.add_argument(
        "--syn-metric", default="mattes", help="Metric for SyN stage (default: mattes)"
    )
    parser.add_argument(
        "--syn-sampling",
        type=int,
        default=32,
        help="Sampling parameter for SyN metric (default: 32)",
    )

    # More complex parameters that need special handling
    parser.add_argument(
        "--reg-iterations", help="SyN iterations, comma-separated (e.g., '40,20,0')"
    )
    parser.add_argument(
        "--aff-iterations",
        help="Affine iterations, comma-separated (e.g., '2100,1200,1200,10')",
    )
    parser.add_argument(
        "--aff-shrink-factors",
        help="Affine shrink factors, comma-separated (e.g., '6,4,2,1')",
    )
    parser.add_argument(
        "--aff-smoothing-sigmas",
        help="Affine smoothing sigmas, comma-separated (e.g., '3,2,1,0')",
    )
    parser.add_argument(
        "--random-seed", type=int, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Process tuple arguments from strings
    kwargs = {}

    if args.verbose:
        kwargs["verbose"] = True

    # Add standard numeric parameters
    for param in [
        "grad_step",
        "flow_sigma",
        "total_sigma",
        "aff_metric",
        "aff_sampling",
        "syn_metric",
        "syn_sampling",
    ]:
        param_value = getattr(args, param.replace("-", "_"))
        if param_value is not None:
            kwargs[param] = param_value

    # Convert comma-separated strings to tuples for complex parameters
    for param in [
        "reg_iterations",
        "aff_iterations",
        "aff_shrink_factors",
        "aff_smoothing_sigmas",
    ]:
        param_value = getattr(args, param.replace("-", "_"))
        if param_value:
            try:
                # Convert string "40,20,0" to tuple (40, 20, 0)
                kwargs[param] = tuple(int(x) for x in param_value.split(","))
            except ValueError:
                print(f"Error parsing {param}. Use comma-separated integers.")
                sys.exit(1)

    # Add random seed if specified
    if args.random_seed is not None:
        kwargs["random_seed"] = args.random_seed

    # Call the coregister function with all arguments
    ants_linear_nonlinear_registration(
        fixed_file=args.fixed,
        moving_file=args.moving,
        out_file=args.output,
        registration_method=args.registration_method,
        affine_file=args.affine_file,
        warp_file=args.warp_file,
        rev_warp_file=args.inverse_warp_file,  # Use standardized name 
        rev_affine_file=args.inverse_affine_file,  # Use standardized name
        initial_affine_file=args.initial_affine_file,
        initial_warp_file=args.initial_warp_file,
        interpolator=args.interpolator,
        threads=args.threads,  # Pass threads parameter
        **kwargs,  # Pass all the extra parameters
    )


if __name__ == "__main__":
    main()
