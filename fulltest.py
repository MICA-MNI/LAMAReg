import os
import csv
import argparse
import numpy as np
import nibabel as nib
import torch
from runtest import (
    run_lamar_registration,
    run_direct_ants_registration,
    run_lamar_registration_robust,
    run_direct_ants_registration_default,
    run_direct_ants_registration_medium_iters,
    run_fsl_registration,  # Add FSL registration import
    compare_registration_quality,
)
from torch_mind import MIND3D


def save_mind_descriptors(moving_img, fixed_img, lamar_output, ants_output, output_dir, lamar_robust_output=None, ants_default_output=None):
    """Extract and save MIND descriptors as multi-channel NIfTI images."""
    print("    Generating MIND descriptor visualizations...")

    # Load images using nibabel
    fixed_nib = nib.load(fixed_img)
    fixed_data = fixed_nib.get_fdata()
    lamar_nib = nib.load(lamar_output)
    lamar_data = lamar_nib.get_fdata()
    ants_nib = nib.load(ants_output)
    ants_data = ants_nib.get_fdata()
    
    # Load optional images
    lamar_robust_data = None
    ants_default_data = None
    if lamar_robust_output:
        lamar_robust_nib = nib.load(lamar_robust_output)
        lamar_robust_data = lamar_robust_nib.get_fdata()
    if ants_default_output:
        ants_default_nib = nib.load(ants_default_output)
        ants_default_data = ants_default_nib.get_fdata()

    # Convert to tensors with batch dimension for MIND
    fixed_tensor = torch.from_numpy(fixed_data).float().unsqueeze(0).unsqueeze(0)
    lamar_tensor = torch.from_numpy(lamar_data).float().unsqueeze(0).unsqueeze(0)
    ants_tensor = torch.from_numpy(ants_data).float().unsqueeze(0).unsqueeze(0)
    
    if lamar_robust_data is not None:
        lamar_robust_tensor = torch.from_numpy(lamar_robust_data).float().unsqueeze(0).unsqueeze(0)
    if ants_default_data is not None:
        ants_default_tensor = torch.from_numpy(ants_default_data).float().unsqueeze(0).unsqueeze(0)

    # Create MIND descriptor
    mind_descriptor = MIND3D(patch_size=3, sigma=0.5)

    # Calculate MIND descriptors
    with torch.no_grad():
        mind_fixed = mind_descriptor(fixed_tensor)
        mind_lamar = mind_descriptor(lamar_tensor)
        mind_ants = mind_descriptor(ants_tensor)
        if lamar_robust_data is not None:
            mind_lamar_robust = mind_descriptor(lamar_robust_tensor)
        if ants_default_data is not None:
            mind_ants_default = mind_descriptor(ants_default_tensor)

    # Create output directory for MIND visualizations
    mind_dir = os.path.join(output_dir, "mind_descriptors")
    os.makedirs(mind_dir, exist_ok=True)

    # Convert from PyTorch tensors to NumPy arrays and save
    fixed_mind_data = np.transpose(mind_fixed[0].numpy(), (1, 2, 3, 0))
    fixed_mind_nii = nib.Nifti1Image(fixed_mind_data, fixed_nib.affine, fixed_nib.header)
    nib.save(fixed_mind_ii, os.path.join(mind_dir, "fixed_mind.nii.gz"))

    lamar_mind_data = np.transpose(mind_lamar[0].numpy(), (1, 2, 3, 0))
    lamar_mind_nii = nib.Nifti1Image(lamar_mind_data, lamar_nib.affine, lamar_nib.header)
    nib.save(lamar_mind_nii, os.path.join(mind_dir, "lamar_mind.nii.gz"))

    ants_mind_data = np.transpose(mind_ants[0].numpy(), (1, 2, 3, 0))
    ants_mind_nii = nib.Nifti1Image(ants_mind_data, ants_nib.affine, ants_nib.header)
    nib.save(ants_mind_nii, os.path.join(mind_dir, "ants_mind.nii.gz"))
    
    # Save optional mind descriptors
    if lamar_robust_data is not None:
        lamar_robust_mind_data = np.transpose(mind_lamar_robust[0].numpy(), (1, 2, 3, 0))
        lamar_robust_mind_nii = nib.Nifti1Image(lamar_robust_mind_data, lamar_robust_nib.affine, lamar_robust_nib.header)
        nib.save(lamar_robust_mind_nii, os.path.join(mind_dir, "lamar_robust_mind.nii.gz"))
        
    if ants_default_data is not None:
        ants_default_mind_data = np.transpose(mind_ants_default[0].numpy(), (1, 2, 3, 0))
        ants_default_mind_nii = nib.Nifti1Image(ants_default_mind_data, ants_default_nib.affine, ants_default_nib.header)
        nib.save(ants_default_mind_nii, os.path.join(mind_dir, "ants_default_mind.nii.gz"))

    print(f"    MIND descriptors saved in {mind_dir}")
    return mind_dir


def normalize_for_vis(array):
    """Normalize array to 0-1 range for visualization."""
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val + 1e-8)


def check_session_completed(output_dir):
    """Check if all registration outputs already exist for this session."""
    required_files = [
        "lamar_registered.nii.gz",
        "lamar_robust_registered.nii.gz",
        "direct_ants_registered.nii.gz", 
        "direct_ants_default_registered.nii.gz",
        "direct_ants_medium_registered.nii.gz",
        "fsl_registered.nii.gz"  # Add FSL output file
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(output_dir, file)):
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch registration between T1w and DWI scans"
    )
    parser.add_argument(
        "--data-path",
        default="/host/verges/tank/data/ian/MICs_MF_Diffusion",
        help="Path to BIDS dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="./registration_results",
        help="Directory for output files and results",
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="Number of threads to use"
    )
    parser.add_argument(
        "--registration-method", default="SyNRA", help="Registration method"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument(
        "--save-mind", action="store_true", help="Save MIND descriptor visualizations"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force reprocessing of completed sessions"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define CSV file path
    results_csv = os.path.join(args.output_dir, "registration_results.csv")
    fieldnames = [
        "subject_id",
        "session_id",
        "lamar_time",
        "lamar_robust_time",
        "ants_time",
        "ants_default_time",
        "ants_medium_time",
        "fsl_time",  # Add FSL time field
        "speedup_lamar_vs_ants",
        "speedup_robust_vs_ants",
        "speedup_lamar_vs_ants_default",
        "speedup_robust_vs_ants_default",
        "speedup_lamar_vs_ants_medium",
        "speedup_robust_vs_ants_medium",
        "speedup_lamar_vs_fsl",  # Add FSL speedup field
        "speedup_robust_vs_fsl",  # Add FSL speedup field
        "mi_lamar",
        "mi_lamar_robust",
        "mi_ants",
        "mi_ants_default",
        "mi_ants_medium",
        "mi_fsl",  # Add FSL metric field
        "antsneighborhoodcorrelation_lamar",
        "antsneighborhoodcorrelation_lamar_robust",
        "antsneighborhoodcorrelation_ants",
        "antsneighborhoodcorrelation_ants_default",
        "antsneighborhoodcorrelation_ants_medium",
        "antsneighborhoodcorrelation_fsl",  # Add FSL metric field
        "mind_lamar",
        "mind_lamar_robust",
        "mind_ants",
        "mind_ants_default", 
        "mind_ants_medium",
        "mind_fsl",  # Add FSL metric field
        "ngf_lamar",
        "ngf_lamar_robust",
        "ngf_ants",
        "ngf_ants_default",
        "ngf_ants_medium",
        "ngf_fsl",  # Add FSL metric field
        "mind_descriptors_dir",
    ]

    # Create CSV file with headers if it doesn't exist
    file_exists = os.path.isfile(results_csv)
    if not file_exists:
        with open(results_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # Iterate through subjects
    for subject_folder in os.listdir(args.data_path):
        subject_dir = os.path.join(args.data_path, subject_folder)
        if not os.path.isdir(subject_dir) or not subject_folder.startswith("sub-"):
            continue

        print(f"Processing subject: {subject_folder}")

        # Iterate through sessions
        for session_folder in os.listdir(subject_dir):
            try:
                session_dir = os.path.join(subject_dir, session_folder)
                if not os.path.isdir(session_dir) or not session_folder.startswith(
                    "ses-"
                ):
                    continue

                print(f"  Processing session: {session_folder}")

                # Find T1w and DWI scans
                t1w_file = None
                dwi_file = None

                # Check for T1w scan
                anatomical_folder = os.path.join(subject_dir, session_folder, "anat")
                if os.path.isdir(anatomical_folder):
                    t1w_path = os.path.join(
                        anatomical_folder,
                        f"{subject_folder}_{session_folder}_T1w-space_T1w.nii.gz",
                    )
                    if os.path.isfile(t1w_path):
                        t1w_file = t1w_path

                # Check for DWI scan
                diffusion_folder = os.path.join(subject_dir, session_folder, "dwi")
                if os.path.isdir(diffusion_folder):
                    dwi_path = os.path.join(
                        diffusion_folder,
                        f"{subject_folder}_{session_folder}_DWI-space_b0.nii.gz",
                    )
                    if os.path.isfile(dwi_path):
                        dwi_file = dwi_path

                # Only proceed if both scans exist
                if t1w_file and dwi_file:
                    print(f"    Found both T1w and DWI scans")

                    # Create subject-specific output directory
                    subj_output_dir = os.path.join(
                        args.output_dir, f"{subject_folder}_{session_folder}"
                    )
                    os.makedirs(subj_output_dir, exist_ok=True)
                    
                    # Define expected output files
                    lamar_output = os.path.join(subj_output_dir, "lamar_registered.nii.gz")
                    lamar_robust_output = os.path.join(subj_output_dir, "lamar_robust_registered.nii.gz")
                    ants_output = os.path.join(subj_output_dir, "direct_ants_registered.nii.gz")
                    ants_default_output = os.path.join(subj_output_dir, "direct_ants_default_registered.nii.gz")
                    ants_medium_output = os.path.join(subj_output_dir, "direct_ants_medium_registered.nii.gz")
                    fsl_output = os.path.join(subj_output_dir, "fsl_registered.nii.gz")  # Add FSL output path
                    
                    # Run LaMAR registration (DWI to T1w) if needed
                    print("    Running LaMAR registration...")
                    lamar_time, lamar_output = run_lamar_registration(
                        moving_img=dwi_file,
                        fixed_img=t1w_file,
                        output_dir=subj_output_dir,
                        registration_method=args.registration_method,
                        threads=args.threads,
                        verbose=not args.quiet,
                        force=args.force,
                    )
                    
                    # Run LaMAR robust registration (DWI to T1w) if needed
                    print("    Running LaMAR registration with robust flag...")
                    lamar_robust_time, lamar_robust_output = run_lamar_registration_robust(
                        moving_img=dwi_file,
                        fixed_img=t1w_file,
                        output_dir=subj_output_dir,
                        registration_method=args.registration_method,
                        threads=args.threads,
                        verbose=not args.quiet,
                        force=args.force,
                    )

                    # Run Direct ANTs registration with custom parameters if needed
                    print("    Running ANTs registration with custom parameters...")
                    ants_time, ants_output = run_direct_ants_registration(
                        moving_img=dwi_file,
                        fixed_img=t1w_file,
                        output_dir=subj_output_dir,
                        registration_method=args.registration_method,
                        threads=args.threads,
                        verbose=not args.quiet,
                        force=args.force,
                    )
                    
                    # Run Direct ANTs registration with default parameters if needed
                    print("    Running ANTs registration with default parameters...")
                    ants_default_time, ants_default_output = run_direct_ants_registration_default(
                        moving_img=dwi_file,
                        fixed_img=t1w_file,
                        output_dir=subj_output_dir,
                        registration_method=args.registration_method,
                        threads=args.threads,
                        verbose=not args.quiet,
                        force=args.force,
                    )
                    
                    # Run Direct ANTs registration with medium iterations (40,20,10) if needed
                    print("    Running ANTs registration with medium iterations (40,20,10)...")
                    ants_medium_time, ants_medium_output = run_direct_ants_registration_medium_iters(
                        moving_img=dwi_file,
                        fixed_img=t1w_file,
                        output_dir=subj_output_dir,
                        registration_method=args.registration_method,
                        threads=args.threads,
                        verbose=not args.quiet,
                        force=args.force,
                    )

                    # Run FSL registration
                    print("    Running FSL registration with BBR...")
                    fsl_time, fsl_output = run_fsl_registration(
                        moving_img=dwi_file,
                        fixed_img=t1w_file,
                        output_dir=subj_output_dir,
                        threads=args.threads,
                        verbose=not args.quiet,
                        force=args.force,
                    )

                    # Compare registration quality for all methods
                    print("    Comparing registration quality...")
                    quality_results = compare_registration_quality(
                        lamar_output=lamar_output,
                        ants_output=ants_output,
                        fixed_img=t1w_file,
                        lamar_robust_output=lamar_robust_output,
                        ants_default_output=ants_default_output,
                        ants_medium_output=ants_medium_output,
                        fsl_output=fsl_output  # Add FSL output
                    )

                    # Calculate speedups including FSL
                    speedup_lamar_vs_ants = ants_time / lamar_time if lamar_time > 0 else 0
                    speedup_robust_vs_ants = ants_time / lamar_robust_time if lamar_robust_time > 0 else 0
                    speedup_lamar_vs_ants_default = ants_default_time / lamar_time if lamar_time > 0 else 0
                    speedup_robust_vs_ants_default = ants_default_time / lamar_robust_time if lamar_robust_time > 0 else 0
                    speedup_lamar_vs_ants_medium = ants_medium_time / lamar_time if lamar_time > 0 else 0
                    speedup_robust_vs_ants_medium = ants_medium_time / lamar_robust_time if lamar_robust_time > 0 else 0
                    speedup_lamar_vs_fsl = fsl_time / lamar_time if lamar_time > 0 else 0
                    speedup_robust_vs_fsl = fsl_time / lamar_robust_time if lamar_robust_time > 0 else 0

                    # Get metrics
                    mi = quality_results.get("mi", {})
                    antsneighborhoodcorrelation = quality_results.get("antsneighborhoodcorrelation", {})
                    mind = quality_results.get("mind", {})
                    ngf = quality_results.get("ngf", {})

                    # Create row data with all methods including FSL
                    row_data = {
                        "subject_id": subject_folder,
                        "session_id": session_folder,
                        "lamar_time": f"{lamar_time:.2f}",
                        "lamar_robust_time": f"{lamar_robust_time:.2f}",
                        "ants_time": f"{ants_time:.2f}",
                        "ants_default_time": f"{ants_default_time:.2f}",
                        "ants_medium_time": f"{ants_medium_time:.2f}",
                        "fsl_time": f"{fsl_time:.2f}",
                        "speedup_lamar_vs_ants": f"{speedup_lamar_vs_ants:.2f}",
                        "speedup_robust_vs_ants": f"{speedup_robust_vs_ants:.2f}",
                        "speedup_lamar_vs_ants_default": f"{speedup_lamar_vs_ants_default:.2f}",
                        "speedup_robust_vs_ants_default": f"{speedup_robust_vs_ants_default:.2f}",
                        "speedup_lamar_vs_ants_medium": f"{speedup_lamar_vs_ants_medium:.2f}",
                        "speedup_robust_vs_ants_medium": f"{speedup_robust_vs_ants_medium:.2f}",
                        "speedup_lamar_vs_fsl": f"{speedup_lamar_vs_fsl:.2f}",
                        "speedup_robust_vs_fsl": f"{speedup_robust_vs_fsl:.2f}",
                        "mi_lamar": f"{mi.get('lamar', 'N/A')}" if mi else "N/A",
                        "mi_lamar_robust": f"{mi.get('lamar_robust', 'N/A')}" if mi else "N/A",
                        "mi_ants": f"{mi.get('ants', 'N/A')}" if mi else "N/A",
                        "mi_ants_default": f"{mi.get('ants_default', 'N/A')}" if mi else "N/A",
                        "mi_ants_medium": f"{mi.get('ants_medium', 'N/A')}" if mi else "N/A",
                        "mi_fsl": f"{mi.get('fsl', 'N/A')}" if mi else "N/A",
                        "antsneighborhoodcorrelation_lamar": f"{antsneighborhoodcorrelation.get('lamar', 'N/A')}" if antsneighborhoodcorrelation else "N/A",
                        "antsneighborhoodcorrelation_lamar_robust": f"{antsneighborhoodcorrelation.get('lamar_robust', 'N/A')}" if antsneighborhoodcorrelation else "N/A",
                        "antsneighborhoodcorrelation_ants": f"{antsneighborhoodcorrelation.get('ants', 'N/A')}" if antsneighborhoodcorrelation else "N/A",
                        "antsneighborhoodcorrelation_ants_default": f"{antsneighborhoodcorrelation.get('ants_default', 'N/A')}" if antsneighborhoodcorrelation else "N/A",
                        "antsneighborhoodcorrelation_ants_medium": f"{antsneighborhoodcorrelation.get('ants_medium', 'N/A')}" if antsneighborhoodcorrelation else "N/A",
                        "antsneighborhoodcorrelation_fsl": f"{antsneighborhoodcorrelation.get('fsl', 'N/A')}" if antsneighborhoodcorrelation else "N/A",
                        "mind_lamar": f"{mind.get('lamar', 'N/A')}" if mind else "N/A",
                        "mind_lamar_robust": f"{mind.get('lamar_robust', 'N/A')}" if mind else "N/A",
                        "mind_ants": f"{mind.get('ants', 'N/A')}" if mind else "N/A",
                        "mind_ants_default": f"{mind.get('ants_default', 'N/A')}" if mind else "N/A",
                        "mind_ants_medium": f"{mind.get('ants_medium', 'N/A')}" if mind else "N/A",
                        "mind_fsl": f"{mind.get('fsl', 'N/A')}" if mind else "N/A",
                        "ngf_lamar": f"{ngf.get('lamar', 'N/A')}" if ngf else "N/A",
                        "ngf_lamar_robust": f"{ngf.get('lamar_robust', 'N/A')}" if ngf else "N/A",
                        "ngf_ants": f"{ngf.get('ants', 'N/A')}" if ngf else "N/A",
                        "ngf_ants_default": f"{ngf.get('ants_default', 'N/A')}" if ngf else "N/A",
                        "ngf_ants_medium": f"{ngf.get('ants_medium', 'N/A')}" if ngf else "N/A",
                        "ngf_fsl": f"{ngf.get('fsl', 'N/A')}" if ngf else "N/A",
                    }

                    # Extract and save MIND descriptors if requested
                    if args.save_mind:
                        mind_dir = save_mind_descriptors(
                            moving_img=dwi_file,
                            fixed_img=t1w_file,
                            lamar_output=lamar_output,
                            ants_output=ants_output,
                            output_dir=subj_output_dir,
                        )
                        row_data["mind_descriptors_dir"] = mind_dir

                    # Write results to CSV by reopening in append mode
                    with open(results_csv, "a", newline="") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow(row_data)

                    # Update print statements
                    print(f"    Quality metrics:")
                    print(f"      MI (LaMAR: {mi.get('lamar', 'N/A')}, LaMAR robust: {mi.get('lamar_robust', 'N/A')}, ANTs: {mi.get('ants', 'N/A')}, ANTs default: {mi.get('ants_default', 'N/A')}, ANTs medium: {mi.get('ants_medium', 'N/A')}, FSL: {mi.get('fsl', 'N/A')})")
                    print(f"      antsneighborhoodcorrelation (LaMAR: {antsneighborhoodcorrelation.get('lamar', 'N/A')}, LaMAR robust: {antsneighborhoodcorrelation.get('lamar_robust', 'N/A')}, ANTs: {antsneighborhoodcorrelation.get('ants', 'N/A')}, ANTs default: {antsneighborhoodcorrelation.get('ants_default', 'N/A')}, ANTs medium: {antsneighborhoodcorrelation.get('ants_medium', 'N/A')}, FSL: {antsneighborhoodcorrelation.get('fsl', 'N/A')})")
                    print(f"      MIND (LaMAR: {mind.get('lamar', 'N/A')}, LaMAR robust: {mind.get('lamar_robust', 'N/A')}, ANTs: {mind.get('ants', 'N/A')}, ANTs default: {mind.get('ants_default', 'N/A')}, ANTs medium: {mind.get('ants_medium', 'N/A')}, FSL: {mind.get('fsl', 'N/A')})")
                    print(f"      NGF (LaMAR: {ngf.get('lamar', 'N/A')}, LaMAR robust: {ngf.get('lamar_robust', 'N/A')}, ANTs: {ngf.get('ants', 'N/A')}, ANTs default: {ngf.get('ants_default', 'N/A')}, ANTs medium: {ngf.get('ants_medium', 'N/A')}, FSL: {ngf.get('fsl', 'N/A')})")
                    
                    print(f"    Completed registration for {subject_folder}_{session_folder}")
                    print(f"    Times: LaMAR: {lamar_time:.2f}s, LaMAR robust: {lamar_robust_time:.2f}s, ANTs: {ants_time:.2f}s, ANTs default: {ants_default_time:.2f}s, ANTs medium: {ants_medium_time:.2f}s, FSL: {fsl_time:.2f}s")
                    print(f"    Speedups vs ANTs: LaMAR: {speedup_lamar_vs_ants:.2f}x, LaMAR robust: {speedup_robust_vs_ants:.2f}x")
                    print(f"    Speedups vs ANTs default: LaMAR: {speedup_lamar_vs_ants_default:.2f}x, LaMAR robust: {speedup_robust_vs_ants_default:.2f}x")
                    print(f"    Speedups vs ANTs medium: LaMAR: {speedup_lamar_vs_ants_medium:.2f}x, LaMAR robust: {speedup_robust_vs_ants_medium:.2f}x")
                    print(f"    Speedups vs FSL: LaMAR: {speedup_lamar_vs_fsl:.2f}x, LaMAR robust: {speedup_robust_vs_fsl:.2f}x")
                    print(f"    Results appended to {results_csv}")
                else:
                    print(f"    Missing T1w or DWI scan, skipping")
            except Exception as e:
                print(f"    Error processing session {session_folder}: {e}")
                continue

    print(f"\nRegistration batch processing complete. Results saved to {results_csv}")


if __name__ == "__main__":
    main()
