"""
apply_warp - Image registration transformation application

Part of the micaflow processing pipeline for neuroimaging data.

This module applies spatial transformations to register images from one space to another
using both affine and non-linear (warp field) transformations. It's commonly used to:
- Transform subject images to a standard space (e.g., MNI152)
- Register images across modalities (e.g., T1w to FLAIR)
- Apply previously calculated transformations to derived images (e.g., segmentations)

The module leverages ANTsPy to apply the transformations in the correct order (warp
field first, then affine) to achieve accurate spatial registration.

API Usage:
---------
micaflow apply_warp
    --moving <path/to/source_image.nii.gz>
    --reference <path/to/target_space.nii.gz>
    --affine <path/to/transform.mat>
    --warp <path/to/warpfield.nii.gz>
    [--output <path/to/registered_image.nii.gz>]

Python Usage:
-----------
>>> import ants
>>> from micaflow.scripts.apply_warp import apply_warp
>>> moving_img = ants.image_read("subject_t1w.nii.gz")
>>> reference_img = ants.image_read("mni152.nii.gz")
>>> apply_warp(
...     moving_img=moving_img,
...     reference_img=reference_img,
...     affine_file="transform.mat",
...     warp_file="warpfield.nii.gz",
...     out_file="registered_t1w.nii.gz"
... )

References:
----------
1. Avants BB, Tustison NJ, Song G, et al. A reproducible evaluation of ANTs
   similarity metric performance in brain image registration. NeuroImage.
   2011;54(3):2033-2044. doi:10.1016/j.neuroimage.2010.09.025
"""

import ants
import argparse
import sys
from colorama import init, Fore, Style

init()


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
    ║                        APPLY WARP                              ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script applies both an affine transformation and a warp field to
    register a moving image to a reference space.
    
    {CYAN}{BOLD}────────────────────────── REQUIRED ARGUMENTS ──────────────────────────{RESET}
      {YELLOW}--moving{RESET}     : Path to the input image to be warped (.nii.gz)
      {YELLOW}--reference{RESET}  : Path to the target/reference image (.nii.gz)
      {YELLOW}--affine{RESET}     : Path to the affine transformation file (.mat)
      {YELLOW}--warp{RESET}       : Path to the warp field (.nii.gz)
    
    {CYAN}{BOLD}────────────────────────── OPTIONAL ARGUMENTS ──────────────────────────{RESET}
      {YELLOW}--output{RESET}     : Output path for the warped image (default: warped_image.nii.gz)
    
    {CYAN}{BOLD}────────────────────────── EXAMPLE USAGE ──────────────────────────{RESET}
    
    {BLUE}# Apply warp transformation{RESET}
    micaflow {GREEN}apply_warp{RESET} {YELLOW}--moving{RESET} subject_t1w.nii.gz {YELLOW}--reference{RESET} mni152.nii.gz \\
      {YELLOW}--affine{RESET} transform.mat {YELLOW}--warp{RESET} warpfield.nii.gz {YELLOW}--output{RESET} registered_t1w.nii.gz
    
    {CYAN}{BOLD}────────────────────────── NOTES ──────────────────────────{RESET}
    {MAGENTA}•{RESET} The order of transforms matters: the warp field is applied first, 
      followed by the affine transformation.
    {MAGENTA}•{RESET} This is the standard order in ANTs for composite transformations.
    """

    print(help_text)


def apply_warp(
    moving_img, reference_img, affine_file, warp_file, out_file, interpolation="linear"
):
    """Apply an affine transform and a warp field to a moving image.

    This function takes a moving image and applies both an affine transformation
    and a nonlinear warp field to register it to a reference image space. The
    transformation is applied using ANTsPy's apply_transforms function with the
    appropriate transform order.

    Parameters
    ----------
    moving_file : str
        Path to the moving image that will be transformed (.nii.gz).
    reference_file : str
        Path to the reference/fixed image that defines the target space (.nii.gz).
    affine_file : str
        Path to the affine transformation file (.mat).
    warp_file : str
        Path to the nonlinear warp field (.nii.gz).
    out_file : str
        Path where the transformed image will be saved.
    interpolation : str, optional
        Interpolation method to use for the transformation. Default is 'linear'.
        Other options include 'nearestNeighbor', 'multiLabel', etc.

    Returns
    -------
    None
        The function saves the transformed image to the specified output path
        but does not return any values.

    Notes
    -----
    The order of transforms matters: the warp field is applied first, followed
    by the affine transformation. This is the standard order in ANTs for
    composite transformations.
    """

    # The order of transforms in transformlist matters (last Transform will be applied first).
    # Usually you put the nonlinear warp first, then the affine:
    transformed = ants.apply_transforms(
        fixed=reference_img,
        moving=moving_img,
        transformlist=[warp_file, affine_file],
        interpolator=interpolation,
    )

    # Save the transformed image
    ants.image_write(transformed, out_file)
    print(f"Saved warped image as {out_file}")


def main():
    # Check if no arguments were provided
    print(len(sys.argv))
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Apply an affine (.mat) and a warp field (.nii.gz) to an image using ANTsPy."
    )
    parser.add_argument(
        "--moving", required=True, help="Path to the moving image (.nii.gz)."
    )
    parser.add_argument(
        "--reference", required=True, help="Path to the reference image (.nii.gz)."
    )
    parser.add_argument(
        "--affine", required=True, help="Path to the affine transform (.mat)."
    )
    parser.add_argument(
        "--warp", required=True, help="Path to the warp field (.nii.gz)."
    )
    parser.add_argument(
        "--output", default="warped_image.nii.gz", help="Output warped image filename."
    )
    parser.add_argument(
        "--interpolation",
        default="linear",
        help="Interpolation method (default: linear).",
    )
    args = parser.parse_args()
    # Load images and transforms
    moving_img = ants.image_read(args.moving)
    reference_img = ants.image_read(args.reference)

    apply_warp(
        moving_img,
        reference_img,
        args.affine,
        args.warp,
        args.output,
        args.interpolation,
    )


if __name__ == "__main__":
    main()
