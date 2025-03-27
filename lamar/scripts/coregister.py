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

init()


def print_help_message():
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
      {YELLOW}--fixed-file{RESET}   : Path to the fixed/reference image (.nii.gz)
      {YELLOW}--moving-file{RESET}  : Path to the moving image to be registered (.nii.gz)
      {YELLOW}--output{RESET}       : Output path for the registered image (.nii.gz)

    {CYAN}{BOLD}─────────────────── OPTIONAL ARGUMENTS ───────────────────{RESET}
      {YELLOW}--warp-file{RESET}      : Path to save the forward warp field (.nii.gz)
      {YELLOW}--affine-file{RESET}    : Path to save the forward affine transform (.mat)
      {YELLOW}--rev-warp-file{RESET}  : Path to save the reverse warp field (.nii.gz)
      {YELLOW}--rev-affine-file{RESET}: Path to save the reverse affine transform (.mat)

    {CYAN}{BOLD}────────────────── EXAMPLE USAGE ────────────────────────{RESET}

    {BLUE}# Register a moving image to a fixed image{RESET}
    micaflow coregister {YELLOW}--fixed-file{RESET} mni152.nii.gz {YELLOW}--moving-file{RESET} subject_t1w.nii.gz \\
      {YELLOW}--output{RESET} registered_t1w.nii.gz {YELLOW}--warp-file{RESET} warp.nii.gz {YELLOW}--affine-file{RESET} affine.mat

    {CYAN}{BOLD}────────────────────────── NOTES ───────────────────────{RESET}
    {MAGENTA}•{RESET} The registration performs SyNRA transformation (rigid+affine+SyN)
    {MAGENTA}•{RESET} Forward transforms convert from moving space to fixed space
    {MAGENTA}•{RESET} Reverse transforms convert from fixed space to moving space
    {MAGENTA}•{RESET} The transforms can be applied to other images using apply_warp
    """
    print(help_text)


def ants_linear_nonlinear_registration(
        fixed_file,
        moving_file,
        out_file="registered_image.nii",
        warp_file=None,
        affine_file=None,
        rev_warp_file=None,
        rev_affine_file=None,
        registration_method="SyNRA"
):
    """Perform linear (rigid + affine) and nonlinear registration using ANTsPy.

    This function performs registration between two images using ANTs' SyNRA transform,
    which includes both linear (rigid + affine) and nonlinear (SyN) components.
    The registered image is saved to the specified output path, and the transform
    files can optionally be saved as well.

    Args:
        fixed_file (str): Path to the fixed/reference image.
        moving_file (str): Path to the moving image that will be registered.
        out_file (str, optional): Path where the registered image will be saved.
            Defaults to "registered_image.nii".
        warp_file (str, optional): Path to save the forward warp field.
            Defaults to None.
        affine_file (str, optional): Path to save the forward affine transform.
            Defaults to None.
        rev_warp_file (str, optional): Path to save the reverse warp field.
            Defaults to None.
        rev_affine_file (str, optional): Path to save the reverse affine transform.
            Defaults to None.

    Returns:
        None: The function saves the registered image and transform files to disk
        but does not return any values.
    """
    # Load images
    fixed = ants.image_read(fixed_file)
    moving = ants.image_read(moving_file)

    # 'SyN' transform includes both linear and nonlinear registration.
    transforms = ants.registration(fixed=fixed, moving=moving, type_of_transform=registration_method)

    # The result of the registration is a dictionary containing, among other keys:
    registered = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=transforms["fwdtransforms"],
                                       interpolator="nearestNeighbor")

    # Save the registered moving image
    ants.image_write(registered, out_file)
    print(f"Registration complete. Saved registered image as {out_file}")

    # If specified, save the transform files
    # Typically, transforms["fwdtransforms"][0] is the warp field, and [1] is the affine.
    if warp_file:
        shutil.copyfile(transforms["fwdtransforms"][0], warp_file)
        print(f"Saved warp field as {warp_file}")
    if affine_file:
        shutil.copyfile(transforms["fwdtransforms"][1], affine_file)
        print(f"Saved affine transform as {affine_file}")
    if rev_warp_file:
        shutil.copyfile(transforms["invtransforms"][1], rev_warp_file)
        print(f"Saved reverse warp field as {rev_warp_file}")
    if rev_affine_file:
        shutil.copyfile(transforms["invtransforms"][0], rev_affine_file)
        print(f"Saved reverse affine transform as {rev_affine_file}")


if __name__ == "__main__":
    # Check if no arguments were provided or help was requested
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Run linear + nonlinear (SyN) registration using ANTsPy."
    )
    parser.add_argument("--fixed-file", required=True, help="Path to the fixed image.")
    parser.add_argument(
        "--moving-file", required=True, help="Path to the moving image."
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for the registered image.",
    )
    parser.add_argument(
        "--warp-file", default=None, help="Optional path to save the warp field."
    )
    parser.add_argument(
        "--affine-file",
        default=None,
        help="Optional path to save the affine transform.",
    )
    parser.add_argument(
        "--rev-warp-file",
        default=None,
        help="Optional path to save the reverse warp field.",
    )
    parser.add_argument(
        "--rev-affine-file",
        default=None,
        help="Optional path to save the reverse affine transform.",
    )
    parser.add_argument(
        "--registration-method",
        default="SyNRA",
        help="Registration method (default: SyNRA)"
    )
    args = parser.parse_args()

    ants_linear_nonlinear_registration(
        args.fixed_file,
        args.moving_file,
        out_file=args.output,
        warp_file=args.warp_file,
        affine_file=args.affine_file,
        rev_warp_file=args.rev_warp_file,
        rev_affine_file=args.rev_affine_file,
        registration_method=args.registration_method
    )