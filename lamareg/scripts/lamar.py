"""
Example script for contrast-agnostic registration using SynthSeg

This script demonstrates a full registration pipeline that uses SynthSeg's brain
parcellation to enable registration between images of different contrasts:

1. Generate parcellations of both input and reference images using SynthSeg
2. Register the parcellations to each other (contrast-agnostic)
3. Apply the resulting transformation to the original input image

This approach is useful for registering images with very different contrasts
(e.g., T1w to T2w, FLAIR to T1w, etc.) where direct intensity-based
registration might fail.
"""

import os
import argparse
import sys
import multiprocessing
import tempfile
import ants
from lamareg.scripts.synthseg import main as synthseg_main
from lamareg.scripts.coregister import ants_linear_nonlinear_registration
from lamareg.scripts.apply_warp import apply_warp

DEFAULT_THREADS = multiprocessing.cpu_count()

def lamareg(
    input_image,
    reference_image,
    output_image=None,
    input_parc=None,
    reference_parc=None,
    output_parc=None,
    generate_warpfield=False,
    apply_warpfield=False,
    registration_method="SyNRA",
    affine_file=None,
    warp_file=None,
    inverse_warp_file=None,
    threads=DEFAULT_THREADS,
    qc_csv=None,
    skip_fixed_parc=False,
    skip_moving_parc=False,
    skip_qc=False,
    disable_robust=False,
    inverse=False,
    secondary_warp_file=None,
    inverse_secondary_warp_file=None,
    verbose=False,
    compose=False
    
):
    """
    Perform contrast-agnostic registration using SynthSeg parcellation.
    """
    temp_files = []
    # Validate arguments based on the selected workflow
    if generate_warpfield and apply_warpfield:
        raise ValueError(
            "Cannot use both --generate-warpfield and --apply-warpfield at the same time"
        )

    # Validate input files exist
    for input_file in [f for f in [input_image, reference_image] if f is not None]:
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

    # Validate thread counts
    if threads < 1:
        threads = DEFAULT_THREADS
    # Workflow-specific validation
    if not apply_warpfield:
        # Registration or Generate-warpfield workflow
        if input_image is None:
            raise ValueError("--moving is required for registration")
        if reference_image is None:
            raise ValueError("--fixed is required for registration")
        # For normal registration (not generate-warpfield), output image is required
        if not generate_warpfield and output_image is None:
            raise ValueError("--output is required for registration")
        
        if skip_qc:
            print("Quality control (QC) generation is disabled")

        if not disable_robust:
            print(
                "\nNote: Robust registration is enabled - a second registration stage will be performed for improved accuracy"
            )
            if compose and warp_file is not None:
                print(
                    "Note: Warp field composition is enabled - warp fields from both stages will be composed"
                )
                print(
                    "Warning: Warpfield composition can result in loss of precision and accuracy in the resulting warpfields. Consider providing a secondary warp field file path for improved results if you plan to re-use warpfields."
                )
                if secondary_warp_file is not None:
                    raise ValueError(
                        "Cannot use --compose with --secondary-warpfield. Choose one or the other."
                    )
                if inverse_secondary_warp_file is not None:
                    raise ValueError(
                        "Cannot use --compose with --inverse-secondary-warpfield. Choose one or the other."
                    )
            if not compose and warp_file is not None:
                print(
                    "Note: Warp field composition is disabled - first and second stage warp fields will be saved separately"
                )
                if secondary_warp_file is None:
                    raise ValueError(
                        "Secondary warp field file path is required when not composing warp fields. Provide --secondary-warpfield."
                    )
                if inverse_secondary_warp_file is None and inverse_warp_file is not None:
                    raise ValueError(
                        "Inverse secondary warp field file path is required when not composing warp fields. Provide --inverse-secondary-warpfield."
                    )
        else:
            print(
                "\nNote: Robust registration is disabled - only a single registration stage will be performed"
            )
            if compose:
                raise ValueError(
                    "Cannot use --compose when robust registration is disabled"
                )
            if secondary_warp_file is not None:
                raise ValueError(
                    "Cannot use --secondary-warpfield when robust registration is disabled"
                )
            if inverse_secondary_warp_file is not None:
                raise ValueError(
                    "Cannot use --inverse-secondary-warpfield when robust registration is disabled"
                )
        # If generating warpfield, warn if transform files not specified
        if affine_file is None:
            print(
                "Warning: No affine transform file path provided - affine transform will not be saved"
            )
            affine_file = tempfile.NamedTemporaryFile(suffix="_tmp_affine.mat", delete=False).name
            temp_files.append(affine_file)
        if warp_file is None:
            print(
                "Warning: No warp field file path provided - warp field will not be saved"
            )
            warp_file = tempfile.NamedTemporaryFile(suffix="_tmp_warp.nii.gz", delete=False).name
            temp_files.append(warp_file)
        if inverse_warp_file is None:
            print(
                "Warning: No inverse warp field file path provided - inverse warp field will not be saved"
            )
        if input_parc is None:
            print(
                "Warning: No moving parcellation file path provided - parcellation will not be saved"
            )
            input_parc = tempfile.NamedTemporaryFile(suffix="_tmp_moving_parc.nii.gz", delete=False).name
            temp_files.append(input_parc)
        if reference_parc is None:
            print(
                "Warning: No fixed parcellation file path provided - parcellation will not be saved"
            )
            reference_parc = tempfile.NamedTemporaryFile(suffix="_tmp_fixed_parc.nii.gz", delete=False).name
            temp_files.append(reference_parc)
        if output_parc is None:
            print(
                "Warning: No output parcellation file path provided - registered parcellation will not be saved"
            )
            output_parc = tempfile.NamedTemporaryFile(suffix="_tmp_registered_parc.nii.gz", delete=False).name
            temp_files.append(output_parc)
    else:
        # Apply-warpfield workflow
        if input_image is None:
            raise ValueError("--moving is required for apply-warpfield")
        if reference_image is None:
            raise ValueError("--fixed is required for apply-warpfield")
        if output_image is None:
            raise ValueError("--output is required for apply-warpfield")
        if affine_file is None and warp_file is None:
            raise ValueError(
                "At least one of --affine or --warpfield is required for apply-warpfield"
            )

        # Validate transform files exist
        for transform_file in [affine_file, warp_file]:
            if not os.path.isfile(transform_file):
                raise FileNotFoundError(f"Transform file not found: {transform_file}")

    # Add QC CSV validation
    if not apply_warpfield and not generate_warpfield:
        if skip_qc:
            pass
        elif qc_csv is None:
            print("Quality control (QC) skipped automatically because --qc-csv was not provided.")
            skip_qc = True
        else:
            qc_dir = os.path.dirname(qc_csv)
            if qc_dir:
                if os.path.exists(qc_dir):
                    if not os.access(qc_dir, os.W_OK):
                        raise PermissionError(
                            f"Cannot write to QC CSV directory: {qc_dir}. Check permissions."
                        )
                else:
                    try:
                        os.makedirs(qc_dir, exist_ok=True)
                    except Exception as e:
                        raise PermissionError(
                            f"Cannot create QC CSV directory: {qc_dir}. Error: {e}"
                        )

    # Create directories for all output files
    for file_path in [
        output_image,
        input_parc,
        reference_parc,
        output_parc,
        affine_file,
        warp_file,
        inverse_warp_file,
        qc_csv,
    ]:
        if file_path is not None:
            output_dir = os.path.dirname(file_path)
            if output_dir:  # Only try to create if there's a directory part
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except PermissionError:
                    raise PermissionError(
                        f"Cannot create output directory: {output_dir}. Check permissions."
                    )

    print(f"Processing input image: {input_image}")
    print(f"Reference image: {reference_image}")
    print(
        f"Using {threads} thread(s)."
    )

    # Set environment variables directly
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)

    temp_warp_file = None
    temp_inverse_warp_file = None
    temp_parc_file = None

    try:
        # WORKFLOW 1 & 2: Full registration or generate warpfield
        if not apply_warpfield:
            # Step 1: Generate parcellations with SynthSeg if needed
            if not skip_moving_parc:
                print("\n--- Step 1.1: Generating parcellation for input image ---")
                synthseg_args = {
                    "i": input_image,
                    "o": input_parc,
                    "parc": True,
                    "robust": True,
                    "fast": False,
                    "cpu": True,
                    "threads": threads,
                    "v1": False,
                    "ct": False,
                    "vol": None,
                    "qc": None,
                    "post": None,
                    "resample": None,
                    "crop": None
                }
                synthseg_main(synthseg_args)
            else:
                print(
                    f"Skipping parcellation generation for input image: {input_image}"
                )
                print(f"Using provided parcellation: {input_parc}")
            # Check if the input parcellation file exists
            if not os.path.isfile(input_parc):
                raise FileNotFoundError(
                    f"Input parcellation file not found: {input_parc}"
                )

            if not skip_fixed_parc:
                print("\n--- Step 1.2: Generating parcellation for reference image ---")
                synthseg_args = {
                    "i": reference_image,
                    "o": reference_parc,
                    "parc": True,
                    "robust": True,
                    "fast": False,
                    "cpu": True,
                    "threads": threads,
                    "v1": False,
                    "ct": False,
                    "vol": None,
                    "qc": None,
                    "post": None,
                    "resample": None,
                    "crop": None
                }
                synthseg_main(synthseg_args)
            else:
                print(
                    f"Skipping parcellation generation for reference image: {reference_image}"
                )
                print(f"Using provided parcellation: {reference_parc}")
            # Check if the reference parcellation file exists
            if not os.path.isfile(reference_parc):
                raise FileNotFoundError(
                    f"Reference parcellation file not found: {reference_parc}"
                )

            # Step 2: Register parcellations using coregister
            print("\n--- Step 2: Coregistering parcellated images ---")
            
            # Determine output file
            current_out_file = output_parc
            if not disable_robust:
                 with tempfile.NamedTemporaryFile(suffix="_tmp_output_parc.nii.gz", delete=False) as tmp_parc:
                    temp_parc_file = tmp_parc.name
                 temp_files.append(temp_parc_file)
                 current_out_file = temp_parc_file
            
            # Determine warp file
            current_warp_file = None
            if warp_file:
                if not disable_robust:
                    if not secondary_warp_file:
                        with tempfile.NamedTemporaryFile(suffix="_tmp_warp.nii.gz", delete=False) as tmp_warp:
                            temp_warp_file = tmp_warp.name
                        temp_files.append(temp_warp_file)
                        current_warp_file = temp_warp_file
                    else:
                        current_warp_file = warp_file
                else:
                    current_warp_file = warp_file

            # Determine inverse warp file
            current_rev_warp_file = None
            if inverse_warp_file:
                if not disable_robust:
                    if not inverse_secondary_warp_file:
                        with tempfile.NamedTemporaryFile(suffix="_tmp_inverse_warp.nii.gz", delete=False) as tmp_inverse_warp:
                            temp_inverse_warp_file = tmp_inverse_warp.name
                        temp_files.append(temp_inverse_warp_file)
                        current_rev_warp_file = temp_inverse_warp_file
                    else:
                        current_rev_warp_file = inverse_warp_file
                else:
                    current_rev_warp_file = inverse_warp_file

            ants_linear_nonlinear_registration(
                fixed_file=reference_parc,
                moving_file=input_parc,
                out_file=current_out_file,
                registration_method=registration_method,
                fixed_image=reference_image,
                affine_file=affine_file,
                warp_file=current_warp_file,
                rev_warp_file=current_rev_warp_file,
                verbose=verbose,
                threads=threads
            )

            if not disable_robust:
                print(
                    "\n--- Step 2.1: Running robust registration for improved accuracy ---"
                )
                
                current_initial_warp = temp_warp_file if not secondary_warp_file else warp_file
                current_initial_inverse_warp = None
                if inverse_warp_file:
                    if not inverse_secondary_warp_file:
                        current_initial_inverse_warp = temp_inverse_warp_file
                    else:
                        current_initial_inverse_warp = inverse_warp_file
                
                current_warp_file = None
                disable_warp_comp = False
                if warp_file or not skip_qc:
                    if secondary_warp_file:
                        current_warp_file = secondary_warp_file
                        disable_warp_comp = True
                    else:
                        current_warp_file = warp_file
                
                current_rev_warp_file = None
                disable_inv_warp_comp = False
                if inverse_warp_file:
                    if inverse_secondary_warp_file:
                        current_rev_warp_file = inverse_secondary_warp_file
                        disable_inv_warp_comp = True
                    else:
                        current_rev_warp_file = inverse_warp_file

                ants_linear_nonlinear_registration(
                    fixed_file=reference_image,
                    moving_file=input_image,
                    interpolator="linear",
                    registration_method="SyNOnly",
                    initial_affine_file=affine_file,
                    initial_warp_file=current_initial_warp,
                    reg_iterations=(10, 20),
                    initial_inverse_warp_file=current_initial_inverse_warp,
                    out_file=None, # Changed from output_image to None to prevent blurred output
                    affine_file=affine_file if (affine_file or not skip_qc) else None,
                    warp_file=current_warp_file,
                    rev_warp_file=current_rev_warp_file,
                    disable_warp_composition=disable_warp_comp,
                    disable_inverse_warp_composition=disable_inv_warp_comp,
                    verbose=verbose,
                    threads=threads
                )

                try:
                    if temp_warp_file and os.path.exists(temp_warp_file):
                        os.remove(temp_warp_file)
                    if temp_inverse_warp_file and os.path.exists(temp_inverse_warp_file):
                        os.remove(temp_inverse_warp_file)
                except OSError:
                    pass

                # Apply warp to parcellation
                moving_img = ants.image_read(input_parc)
                reference_img = ants.image_read(reference_image)
                
                apply_warp(
                    moving_img=moving_img,
                    reference_img=reference_img,
                    affine_file=affine_file,
                    warp_file=warp_file,
                    out_file=output_parc,
                    interpolation="nearestNeighbor",
                    secondary_warp=secondary_warp_file,
                    verbose=verbose
                )
                    
            # Run Dice evaluation after coregistration
            if not skip_qc:


                # If qc_csv is not provided, generate a default path based on output_parc
                dice_output = (
                    qc_csv
                    if qc_csv
                    else output_parc.split('.')[0] + "_dice_scores.csv"
                )

                print(
                    "\n--- Step 3: Calculating Dice scores to evaluate registration quality ---"
                )
                try:
                    from lamareg.scripts.dice_compare import compare_parcellations_dice

                    compare_parcellations_dice(reference_parc, output_parc, dice_output, verbose=verbose)
                    print(f"Quality control metrics saved to: {dice_output}")
                except FileNotFoundError as e:
                    print(
                        f"Warning: Could not calculate Dice scores - file not found: {e}",
                        file=sys.stderr,
                    )
                except PermissionError as e:
                    print(
                        f"Warning: Could not calculate Dice scores - permission error: {e}",
                        file=sys.stderr,
                    )
                except ImportError as e:
                    print(
                        f"Warning: Could not calculate Dice scores - dice_compare module not found",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(
                        f"Warning: Could not calculate Dice scores: {e}",
                        file=sys.stderr,
                    )

        # WORKFLOW 1 & 3: Apply transformation to the original input image
        # Run this step for ALL registration cases if an output image is requested
        # Previously this was restricted: (disable_robust or apply_warpfield)
        if not generate_warpfield and output_image is not None:
            print(
                "\n--- Step 4: Applying transformation to original input image ---"
            )
            
            moving_img = ants.image_read(input_image)
            reference_img = ants.image_read(reference_image)

            apply_warp(
                moving_img=moving_img,
                reference_img=reference_img,
                affine_file=affine_file,
                warp_file=warp_file,
                out_file=output_image,
                secondary_warp=secondary_warp_file,
                inverse=inverse,
                verbose=verbose
            )

            print(f"\nSuccess! Registered image saved to: {output_image}")
        elif generate_warpfield:
            success_msg = "\nSuccess! "
            if warp_file:
                success_msg += f"Warp field generated at: {warp_file}"
            if affine_file:
                success_msg += f"\nAffine transformation saved at: {affine_file}"
            print(success_msg)

    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        for temp_path in temp_files:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


def main():
    """Entry point for command-line use"""
    parser = argparse.ArgumentParser(
        description="Contrast-agnostic registration using SynthSeg"
    )
    parser.add_argument(
        "--moving", required=True, help="Input moving image to be registered"
    )
    parser.add_argument(
        "--fixed", required=True, help="Reference fixed image (target space)"
    )
    parser.add_argument(
        "--skip-fixed-parc",
        action="store_true",
        help="Skip generating fixed parcellation",
    )
    parser.add_argument(
        "--skip-moving-parc",
        action="store_true",
        help="Skip generating moving parcellation",
    )
    parser.add_argument("--output", required=True, help="Output registered image")
    parser.add_argument(
        "--moving-parc", help="Path for moving image parcellation"
    )
    parser.add_argument(
        "--fixed-parc", help="Path for fixed image parcellation"
    )
    parser.add_argument(
        "--registered-parc", help="Path for registered parcellation"
    )
    parser.add_argument(
        "--affine", help="Path for affine transformation"
    )
    parser.add_argument("--warpfield", help="Path for warp field")
    parser.add_argument("--inverse-warpfield", help="Path for inverse warp field")
    parser.add_argument(
        "--generate-warpfield",
        action="store_true",
        help="Generate warp field without applying it",
    )
    parser.add_argument(
        "--apply-warpfield",
        action="store_true",
        help="Apply existing warp field to moving image",
    )
    parser.add_argument(
        "--registration-method", default="SyNRA", help="Registration method"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="Number of threads to use for processing",
    )

    parser.add_argument("--secondary-warpfield", help="Path for secondary warp field (for robust registration)")
    parser.add_argument("--inverse-secondary-warpfield", help="Path for inverse of secondary warp field (for robust registration)")
    parser.add_argument("--qc-csv", help="Path for quality control Dice score CSV file")
    parser.add_argument("--skip-qc", action="store_true", help="Skip QC CSV generation")
    parser.add_argument("--disable-robust", action="store_true", help="Disable robust second-stage registration")
    parser.add_argument("--inverse", action="store_true", help="Whether to reverse the order of the transforms (warpfield first, then affine)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--compose", action="store_true", help="Compose warp fields instead of applying sequentially")
    args = parser.parse_args()

    # Validate arguments based on workflow
    if args.apply_warpfield and (args.affine is None or args.warpfield is None):
        parser.error("--apply-warpfield requires --affine and --warpfield arguments")

    if args.generate_warpfield and args.output is not None:
        parser.error(
            "--generate-warpfield cannot be used with --output (no output image is produced)"
        )

    if not args.apply_warpfield and not args.generate_warpfield and args.output is None:
        parser.error(
            "--output is required unless --generate-warpfield or --apply-warpfield is specified"
        )

    lamareg(
        input_image=args.moving,
        reference_image=args.fixed,
        output_image=args.output,
        input_parc=args.moving_parc,
        reference_parc=args.fixed_parc,
        output_parc=args.registered_parc,
        generate_warpfield=args.generate_warpfield,
        apply_warpfield=args.apply_warpfield,
        registration_method=args.registration_method,
        affine_file=args.affine,
        warp_file=args.warpfield,
        inverse_warp_file=args.inverse_warpfield,
        threads=args.threads,
        qc_csv=args.qc_csv,
        skip_fixed_parc=args.skip_fixed_parc,
        skip_moving_parc=args.skip_moving_parc,
        skip_qc=args.skip_qc,
        disable_robust=args.disable_robust,
        inverse=args.inverse,
        secondary_warp_file=args.secondary_warpfield,
        inverse_secondary_warp_file=args.inverse_secondary_warpfield,
        verbose=args.verbose,
        compose=args.compose
    )


if __name__ == "__main__":
    main()
