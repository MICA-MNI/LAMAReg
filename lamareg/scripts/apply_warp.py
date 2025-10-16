"""
apply_warp - Image registration transformation application

Part of the LAMAR processing pipeline for neuroimaging data.

This module applies spatial transformations to register images from one space to another
using both affine and non-linear (warp field) transformations. It's commonly used to:
- Transform subject images to a standard space (e.g., MNI152)
- Register images across modalities (e.g., T1w to FLAIR)
- Apply previously calculated transformations to derived images (e.g., segmentations)

The module leverages ANTsPy to apply the transformations in the correct order.

Transform Order:
---------------
By default (inverse=False), transforms are applied in this order:
    moving → [secondary_warp] → [warp] → [affine] → fixed

When inverse=True, the order is reversed:
    moving → [affine^-1] → [warp] → [secondary_warp] → fixed

Note: In ANTs' transformlist, the LAST transform in the list is applied FIRST.
Therefore, to achieve moving → warp → affine, we provide [warp, affine] to ANTs.

Robust Mode (Two Warpfields):
-----------------------------
When using LAMAR's robust registration mode, you'll have two warpfields:
1. Primary warpfield: From parcellation-based registration (coarse alignment)
2. Secondary warpfield: From direct image registration (fine-tuning)

To apply both, use:
    --warp primary_warp.nii.gz --secondary-warp secondary_warp.nii.gz

The composition order is: moving → secondary → primary → affine → fixed

API Usage:
---------
lamar apply-warp \\
    --moving <path/to/source_image.nii.gz> \\
    --fixed <path/to/target_space.nii.gz> \\
    --affine <path/to/transform.mat> \\
    --warp <path/to/warpfield.nii.gz> \\
    [--secondary-warp <path/to/secondary_warpfield.nii.gz>] \\
    [--output <path/to/registered_image.nii.gz>] \\
    [--interpolation linear|nearestNeighbor|multiLabel] \\
    [--inverse]

Python Usage:
-----------
>>> import ants
>>> from lamareg.scripts.apply_warp import apply_warp
>>> moving_img = ants.image_read("subject_t1w.nii.gz")
>>> reference_img = ants.image_read("mni152.nii.gz")
>>> apply_warp(
...     moving_img=moving_img,
...     reference_img=reference_img,
...     affine_file="transform.mat",
...     warp_file="warpfield.nii.gz",
...     secondary_warp="secondary_warp.nii.gz",
...     out_file="registered_t1w.nii.gz",
...     interpolation="linear"
... )

Examples:
--------
# Apply single warpfield + affine (single-stage registration):
lamar apply-warp \\
    --moving subject_flair.nii.gz \\
    --fixed subject_t1w.nii.gz \\
    --warp flair_to_t1w_warp.nii.gz \\
    --affine flair_to_t1w_affine.mat \\
    --output registered_flair.nii.gz

# Apply two warpfields + affine (robust mode):
lamar apply-warp \\
    --moving dwi_segmentation.nii.gz \\
    --fixed T1w_reference.nii.gz \\
    --warp dwi_to_T1w_primary_warp.nii.gz \\
    --secondary-warp dwi_to_T1w_secondary_warp.nii.gz \\
    --affine dwi_to_T1w_affine.mat \\
    --output dwi_seg_in_T1w.nii.gz

# Apply inverse transforms (from fixed back to moving space):
lamar apply-warp \\
    --moving atlas_in_T1w.nii.gz \\
    --fixed subject_dwi.nii.gz \\
    --warp T1w_to_dwi_warp.nii.gz \\
    --affine T1w_to_dwi_affine.mat \\
    --output atlas_in_dwi.nii.gz \\
    --inverse

# Apply to label image (use nearest neighbor interpolation):
lamar apply-warp \\
    --moving segmentation.nii.gz \\
    --fixed mni152.nii.gz \\
    --warp warp.nii.gz \\
    --affine affine.mat \\
    --output seg_in_mni.nii.gz \\
    --interpolation nearestNeighbor

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
    
    This script applies spatial transformations (affine + warp fields) to
    register a moving image to a reference space.
    
    {CYAN}{BOLD}────────────────────────── REQUIRED ARGUMENTS ──────────────────────────{RESET}
      {YELLOW}--moving{RESET}     : Path to the input image to be warped (.nii.gz)
      {YELLOW}--fixed{RESET}      : Path to the target/reference image (.nii.gz)
    
    {CYAN}{BOLD}────────────────────────── OPTIONAL ARGUMENTS ──────────────────────────{RESET}
      {YELLOW}--affine{RESET}            : Path to the affine transformation file (.mat)
      {YELLOW}--warp{RESET}              : Path to the primary warp field (.nii.gz)
      {YELLOW}--secondary-warp{RESET}    : Path to the secondary warp field (.nii.gz) for robust mode
      {YELLOW}--output{RESET}            : Output path for the warped image (default: warped_image.nii.gz)
      {YELLOW}--interpolation{RESET}     : Interpolation method (default: linear)
                             Options: linear, nearestNeighbor, multiLabel, bSpline
      {YELLOW}--inverse{RESET}           : Apply inverse transforms (fixed → moving direction)
    
    {CYAN}{BOLD}────────────────────────── TRANSFORM ORDER ──────────────────────────{RESET}
    
    {BLUE}Forward (default):{RESET}
      moving → {MAGENTA}[secondary_warp]{RESET} → {MAGENTA}[warp]{RESET} → {MAGENTA}[affine]{RESET} → fixed
    
    {BLUE}Inverse (--inverse flag):{RESET}
      fixed → {MAGENTA}[affine^-1]{RESET} → {MAGENTA}[warp]{RESET} → {MAGENTA}[secondary_warp]{RESET} → moving
    
    {CYAN}{BOLD}────────────────────────── EXAMPLE USAGE ──────────────────────────{RESET}
    
    {BLUE}# Single-stage registration (warp + affine):{RESET}
    lamar {GREEN}apply-warp{RESET} {YELLOW}--moving{RESET} subject_flair.nii.gz {YELLOW}--fixed{RESET} subject_t1w.nii.gz \\
      {YELLOW}--warp{RESET} flair_to_t1w_warp.nii.gz {YELLOW}--affine{RESET} flair_to_t1w_affine.mat \\
      {YELLOW}--output{RESET} registered_flair.nii.gz
    
    {BLUE}# Robust mode (two warpfields + affine):{RESET}
    lamar {GREEN}apply-warp{RESET} {YELLOW}--moving{RESET} dwi.nii.gz {YELLOW}--fixed{RESET} T1w.nii.gz \\
      {YELLOW}--warp{RESET} dwi_to_T1w_primary.nii.gz \\
      {YELLOW}--secondary-warp{RESET} dwi_to_T1w_secondary.nii.gz \\
      {YELLOW}--affine{RESET} dwi_to_T1w_affine.mat \\
      {YELLOW}--output{RESET} dwi_in_T1w.nii.gz
    
    {BLUE}# Apply to segmentation (nearest neighbor interpolation):{RESET}
    lamar {GREEN}apply-warp{RESET} {YELLOW}--moving{RESET} parcellation.nii.gz {YELLOW}--fixed{RESET} mni152.nii.gz \\
      {YELLOW}--warp{RESET} warp.nii.gz {YELLOW}--affine{RESET} affine.mat \\
      {YELLOW}--interpolation{RESET} nearestNeighbor \\
      {YELLOW}--output{RESET} parc_in_mni.nii.gz
    
    {BLUE}# Apply inverse transforms:{RESET}
    lamar {GREEN}apply-warp{RESET} {YELLOW}--moving{RESET} atlas_in_T1w.nii.gz {YELLOW}--fixed{RESET} dwi.nii.gz \\
      {YELLOW}--warp{RESET} T1w_to_dwi_warp.nii.gz {YELLOW}--affine{RESET} T1w_to_dwi_affine.mat \\
      {YELLOW}--inverse{RESET} {YELLOW}--output{RESET} atlas_in_dwi.nii.gz
    
    {CYAN}{BOLD}────────────────────────── NOTES ───────────────────────────{RESET}
    
    {MAGENTA}•{RESET} At least one transform (--affine, --warp, or --secondary-warp) must be provided
    {MAGENTA}•{RESET} For label/segmentation images, use --interpolation nearestNeighbor or multiLabel
    {MAGENTA}•{RESET} For continuous images (T1w, FLAIR, etc.), use --interpolation linear or bSpline
    {MAGENTA}•{RESET} The --inverse flag inverts the affine but applies warpfields as-is
    {MAGENTA}•{RESET} Robust mode requires both --warp and --secondary-warp from LAMAR registration
    """

    print(help_text)


def apply_warp(
    moving_img, reference_img, affine_file, warp_file, out_file, interpolation="linear", inverse=False, secondary_warp=None, verbose=False
):
    """Apply affine and warp field transformations to a moving image.

    This function takes a moving image and applies spatial transformations to register
    it to a reference image space. It supports both single-stage registration (one warp)
    and robust two-stage registration (primary + secondary warp).

    Parameters
    ----------
    moving_img : ants.ANTsImage
        The moving image that will be transformed.
    reference_img : ants.ANTsImage
        The reference/fixed image that defines the target space.
    affine_file : str or None
        Path to the affine transformation file (.mat). Can be None if not needed.
    warp_file : str or None
        Path to the primary nonlinear warp field (.nii.gz). Can be None if not needed.
    out_file : str
        Path where the transformed image will be saved.
    interpolation : str, optional
        Interpolation method to use for the transformation. Default is 'linear'.
        Options: 'linear', 'nearestNeighbor', 'multiLabel', 'bSpline', 'gaussian', etc.
    inverse : bool, optional
        If True, reverses the transform order and inverts the affine. Default is False.
    secondary_warp : str or None, optional
        Path to the secondary warp field (.nii.gz) for robust mode. Default is None.

    Returns
    -------
    None
        The function saves the transformed image to the specified output path.

    Notes
    -----
    Transform Order (inverse=False):
        moving → [secondary_warp] → [warp] → [affine] → fixed
    
    Transform Order (inverse=True):
        moving → [affine^-1] → [warp] → [secondary_warp] → fixed
    
    ANTs applies transforms in reverse order of the transformlist, so the last
    transform in the list is applied first. This function handles the ordering
    automatically based on the inverse flag.
    
    For robust mode (two-stage registration), both warp_file and secondary_warp
    should be provided. The secondary warp is applied first, followed by the
    primary warp, then the affine.

    Examples
    --------
    >>> import ants
    >>> from lamareg.scripts.apply_warp import apply_warp
    >>> moving = ants.image_read("dwi.nii.gz")
    >>> fixed = ants.image_read("T1w.nii.gz")
    >>> 
    >>> # Single-stage registration
    >>> apply_warp(moving, fixed, "affine.mat", "warp.nii.gz", "output.nii.gz")
    >>> 
    >>> # Robust mode (two warpfields)
    >>> apply_warp(moving, fixed, "affine.mat", "primary.nii.gz", "output.nii.gz",
    ...            secondary_warp="secondary.nii.gz")
    >>> 
    >>> # Apply to segmentation
    >>> apply_warp(moving, fixed, "affine.mat", "warp.nii.gz", "seg_out.nii.gz",
    ...            interpolation="nearestNeighbor")
    """

    # The order of transforms in transformlist matters (last transform will be applied first).
    # Usually you put the nonlinear warp first, then the affine:
    if not (affine_file or warp_file or secondary_warp):
        print("ERROR: At least one of --affine, --warp, or --secondary-warp must be provided.")
        sys.exit(1)
    
    if inverse:
        # Inverse order: affine^-1, then warp, then secondary_warp
        transformlist = []
        invertlist = []
        if affine_file:
            transformlist.append(affine_file)
            invertlist.append(True)  # Invert the affine
        if warp_file:
            transformlist.append(warp_file)
            invertlist.append(False)  # Don't invert warp (already in inverse direction)
        if secondary_warp:
            transformlist.append(secondary_warp)
            invertlist.append(False)  # Don't invert secondary warp

        transformed = ants.apply_transforms(
            fixed=reference_img,
            moving=moving_img,
            transformlist=transformlist,
            interpolator=interpolation,
            whichtoinvert=invertlist
        )
    else:
        # Forward order: secondary_warp, then warp, then affine
        # ANTs applies in reverse, so list them backwards
        transformlist = []
        if secondary_warp:
            transformlist.append(secondary_warp)
        if warp_file:
            transformlist.append(warp_file)
        if affine_file:
            transformlist.append(affine_file)
        
        transformed = ants.apply_transforms(
            fixed=reference_img,
            moving=moving_img,
            transformlist=transformlist,
            interpolator=interpolation,
        )

    # Save the transformed image
    ants.image_write(transformed, out_file)
    print(f"Saved warped image as {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply affine and warp field transformations to an image using ANTsPy.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--moving", required=True, help="Path to the moving image (.nii.gz)."
    )
    parser.add_argument(
        "--fixed", required=True, help="Path to the fixed/reference image (.nii.gz)."
    )
    parser.add_argument(
        "--output", default="warped_image.nii.gz", help="Output warped image filename."
    )
    parser.add_argument(
        "--affine", help="Path to the affine transform (.mat)."
    )
    parser.add_argument(
        "--warp", help="Path to the primary warp field (.nii.gz)."
    )
    parser.add_argument(
        "--secondary-warp", help="Path to the secondary warp field (.nii.gz) for robust mode."
    )
    parser.add_argument(
        "--interpolation",
        default="linear",
        help="Interpolation method (default: linear). Options: linear, nearestNeighbor, multiLabel, bSpline."
    )
    parser.add_argument(
        "--inverse",
        action="store_true",
        help="Apply inverse transforms (fixed → moving direction, inverts affine)"
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed output to console")
    args = parser.parse_args()

    moving_img = ants.image_read(args.moving)
    reference_img = ants.image_read(args.fixed)

    apply_warp(
        moving_img,
        reference_img,
        args.affine,
        args.warp,
        args.output,
        args.interpolation,
        inverse=args.inverse,
        secondary_warp=args.secondary_warp if hasattr(args, 'secondary_warp') else None,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
