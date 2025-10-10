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
from scipy.interpolate import RegularGridInterpolator
from concurrent.futures import ThreadPoolExecutor
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

def compose_3d_fields_extrap(A, B):
    """
    Compose displacement fields with linear interpolation + extrapolation.
    A, B: (X, Y, Z, 3) in voxel units. Returns C of same shape.
    """
    if A.shape != B.shape or A.shape[-1] != 3:
        raise ValueError("A and B must be (X, Y, Z, 3)")

    X, Y, Z, _ = A.shape
    i = np.arange(X, dtype=np.float32)
    j = np.arange(Y, dtype=np.float32)
    k = np.arange(Z, dtype=np.float32)

    # Base grid and warped coords
    I, J, K = np.meshgrid(i, j, k, indexing='ij')
    Iw = I + A[..., 0]
    Jw = J + A[..., 1]
    Kw = K + A[..., 2]
    pts = np.column_stack([Iw.ravel(), Jw.ravel(), Kw.ravel()])

    C = np.empty_like(A)
    
    def interpolate_component(c):
        rgi = RegularGridInterpolator(
            (i, j, k), B[..., c],
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        Bw = rgi(pts).reshape(X, Y, Z)
        return c, A[..., c] + Bw
    
    # Use ThreadPoolExecutor for parallel interpolation
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(interpolate_component, range(3))
    
    for c, result in results:
        C[..., c] = result
    
    return C

def mm_to_vox_field(D_mm, affine):
    """
    Convert a displacement field in mm to voxel units.
    D_mm: (X,Y,Z,3) in mm; affine: (4,4) of the image grid.
    """
    # Use the linear part (rotation+scaling) to map vectors; ignore translation
    M = affine[:3, :3]
    Minv = np.linalg.inv(M)
    # Apply Minv to vectors (broadcast over grid)
    D_vox = np.tensordot(D_mm, Minv.T, axes=([3],[0]))  # -> (X,Y,Z,3)
    return D_vox

def vox_to_mm_field(D_vox, affine):
    M = affine[:3, :3]
    D_mm = np.tensordot(D_vox, M.T, axes=([3],[0]))
    return D_mm

def compose_warps(warp1_path, warp2_path, out_path):
    # Load two vector fields (NIfTI) with last dim=3
    A_img = nib.load(warp2_path)
    B_img = nib.load(warp1_path)
    A_mm = A_img.get_fdata(dtype=np.float32).squeeze()  # (X,Y,Z,3)
    B_mm = B_img.get_fdata(dtype=np.float32).squeeze()

    # Convert mm -> voxel units
    A_vox = mm_to_vox_field(A_mm, A_img.affine)
    B_vox = mm_to_vox_field(B_mm, B_img.affine)

    # Compose
    C_vox = compose_3d_fields_extrap(A_vox, B_vox)

    # Back to mm (optional)
    C_mm = vox_to_mm_field(C_vox, A_img.affine)

    # Ensure channel-last 3 components
    if C_mm.ndim == 4 and C_mm.shape[-1] == 3:
        pass
    elif C_mm.ndim == 4 and C_mm.shape[0] == 3:
        C_mm = np.moveaxis(C_mm, 0, -1)
    else:
        raise ValueError(f"Expected (X,Y,Z,3) or (3,X,Y,Z), got {C_mm.shape}")

    C_mm = np.ascontiguousarray(C_mm)

    X, Y, Z, _ = C_mm.shape
    # Make (X,Y,Z,1,3)
    data5 = C_mm[..., None, :]            # add a singleton 4th dim -> (X,Y,Z,1,3)

    hdr5 = nib.Nifti1Header()
    hdr5.set_data_dtype(np.float32)
    hdr5.set_xyzt_units('mm','sec')
    hdr5.set_intent('vector')

    hdr5['dim'][0] = 5                    # 5D image
    hdr5['dim'][1] = X
    hdr5['dim'][2] = Y
    hdr5['dim'][3] = Z
    hdr5['dim'][4] = 1
    hdr5['dim'][5] = 3                    # components = 3
    hdr5['pixdim'][4] = 1.0               # spacing for dim4
    hdr5['pixdim'][5] = 1.0               # spacing for dim5 (components), arbitrary

    img5 = nib.Nifti1Image(data5, affine=A_img.affine, header=hdr5)
    nib.save(img5, out_path)

def ants_linear_nonlinear_registration(
    fixed_file,
    moving_file,
    out_file=None,
    warp_file=None,
    affine_file=None,
    rev_warp_file=None,
    registration_method="SyNRA",
    initial_affine_file=None,
    initial_warp_file=None,
    interpolator="genericLabel",
    threads=DEFAULT_THREADS,
    fixed_image=None,
    initial_inverse_warp_file=None,
    **kwargs,
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
    ):
        print(Fore.RED + "Error: No outputs specified." + Style.RESET_ALL)
        sys.exit(1)

    # Set ANTs/ITK thread count in environment variables
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)

    # Load images
    fixed = ants.image_read(fixed_file)
    moving = ants.image_read(moving_file)
    
    # Track temporary files for cleanup
    temp_files_to_cleanup = []
    
    # If initial transforms are provided, apply them first and register the warped image
    if initial_warp_file or initial_affine_file:
        print("Applying initial transforms to moving image...")
        
        # Build initial transform list (order matters: warp then affine)
        initial_transformlist = []
        if initial_warp_file:
            initial_transformlist.append(initial_warp_file)
        if initial_affine_file:
            initial_transformlist.append(initial_affine_file)
        
        # Apply initial transforms to moving image
        warped_moving = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=initial_transformlist,
            interpolator=interpolator,
        )
        
        # Save temporarily warped moving image using tempfile
        temp_fd, temp_warped_file = tempfile.mkstemp(suffix='.nii.gz', prefix='temp_warped_moving_')
        os.close(temp_fd)  # Close the file descriptor as we'll write with ANTs
        ants.image_write(warped_moving, temp_warped_file)
        temp_files_to_cleanup.append(temp_warped_file)
        
        print(f"Initial transforms applied. Saved temporary warped image as {temp_warped_file}")
        print("Now performing new registration on pre-warped image...")
        
        # Register the warped moving image to fixed
        transforms = ants.registration(
            fixed=fixed,
            moving=warped_moving,
            type_of_transform=registration_method,
            initial_transform=None,  # No initial transform for this registration
            **kwargs,
        )
    else:
        # No initial transforms, just perform standard registration
        transforms = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=registration_method,
            initial_transform=None,
            **kwargs,
        )
    
    
    # Save the registered moving image
    if out_file is not None:
        ants.image_write(transforms['warpedmovout'], out_file)
        print(f"Registration complete. Saved registered image as {out_file}")

    # Save forward warp
    if warp_file:
        if initial_warp_file:
            # Compose the new warp with the initial warp
            compose_warps(
                initial_warp_file,
                transforms["fwdtransforms"][0],
                warp_file
            )
            print(f"Composed and saved warp field as {warp_file}")
        else:
            if fixed_image:
                # Resample the warp field to the moving image space
                moving_img = ants.image_read(fixed_image)
                warp_field = ants.image_read(transforms["fwdtransforms"][0])
                
                components = []
                for i in range(warp_field.components):
                    component = ants.split_channels(warp_field)[i]
                    resampled_comp = ants.resample_image_to_target(
                        component, 
                        moving_img,
                        interp_type='linear'
                    )
                    components.append(resampled_comp)
                
                resampled_warp = ants.merge_channels(components)
                ants.image_write(resampled_warp, warp_file)
                print(f"Resampled and saved warp field to match moving image space as {warp_file}")
            else:
                shutil.copyfile(transforms["fwdtransforms"][0], warp_file)
                print(f"Saved warp field as {warp_file}")

    # Save forward affine
    if affine_file:
        if initial_affine_file:
            # The new affine is at index 1, compose it with initial affine
            # Note: ANTs composition happens automatically when applying transforms
            shutil.copyfile(transforms["fwdtransforms"][1], affine_file)
            print(f"Saved affine transform as {affine_file}")
        else:
            shutil.copyfile(transforms["fwdtransforms"][1], affine_file)
            print(f"Saved affine transform as {affine_file}")
    
    # Save inverse warp
    if rev_warp_file:
        if initial_warp_file:
            # Compose the inverse warps
            compose_warps(
                transforms["invtransforms"][1],
                initial_inverse_warp_file,
                rev_warp_file
            )
            print(f"Composed and saved inverse warp field as {rev_warp_file}")
        else:
            if fixed_image:
                fixed_img = ants.image_read(fixed_image)
                warp_field = ants.image_read(transforms["invtransforms"][1])
                
                components = []
                for i in range(warp_field.components):
                    component = ants.split_channels(warp_field)[i]
                    resampled_comp = ants.resample_image_to_target(
                        component, 
                        fixed_img,
                        interp_type='linear'
                    )
                    components.append(resampled_comp)
                
                resampled_warp = ants.merge_channels(components)
                ants.image_write(resampled_warp, rev_warp_file)
                print(f"Resampled and saved inverse warp field to match fixed image space as {rev_warp_file}")
            else:
                shutil.copyfile(transforms["invtransforms"][1], rev_warp_file)
                print(f"Saved warp field as {rev_warp_file}")
    


    print("All specified outputs saved successfully.")
    
    # Clean up temporary files
    print("Cleaning up temporary files...")
    temp_files_to_delete = set(transforms['fwdtransforms'] + transforms['invtransforms'] + temp_files_to_cleanup)
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
    parser.add_argument(
        "--fixed-image", help="Original moving image, if provided, the warpfield will be resampled to this space."
    )
    parser.add_argument(
        "--initial-inverse-warp-file", help="Initial inverse warp field file path"
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
    parser.add_argument(
        "--total-sigma",
        type=float,
        default=0,
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
        "--aff-metric", default="mattes", help="Metric for affine stage (default: mattes)"
    )
    parser.add_argument("--aff-sampling", type=int, default=32, help="Sampling parameter for affine metric (default: 32)")
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
        initial_affine_file=args.initial_affine_file,
        initial_warp_file=args.initial_warp_file,
        interpolator=args.interpolator,
        threads=args.threads,  # Pass threads parameter
        fixed_image=args.fixed_image,
        initial_inverse_warp_file=args.initial_inverse_warp_file,
        **kwargs,  # Pass all the extra parameters
    )


if __name__ == "__main__":
    main()
