import os
import csv
import argparse
import numpy as np
import nibabel as nib
import torch
import gc
from runtest import (
    run_lamar_registration,
    run_direct_ants_registration,
    run_lamar_registration_robust,
    run_direct_ants_registration_default,
    run_direct_ants_registration_medium_iters,
    run_fsl_registration,
    run_easyreg_registration,  # Add this import
    compare_registration_quality,
)
from torch_mind import MIND3D


def save_mind_descriptors(moving_img, fixed_img, lamar_output, ants_output, output_dir, 
                         lamar_robust_output=None, ants_default_output=None,
                         ants_medium_output=None, fsl_output=None, easyreg_output=None):
    """Extract and save MIND descriptors as multi-channel NIfTI images."""
    print("    Generating MIND descriptor visualizations...")

    try:
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
        lamar_robust_nib = None
        ants_default_nib = None
        
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
        
        lamar_robust_tensor = None
        ants_default_tensor = None
        
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
            
            mind_lamar_robust = None
            mind_ants_default = None
            
            if lamar_robust_tensor is not None:
                mind_lamar_robust = mind_descriptor(lamar_robust_tensor)
            if ants_default_tensor is not None:
                mind_ants_default = mind_descriptor(ants_default_tensor)

        # Create output directory for MIND visualizations
        mind_dir = os.path.join(output_dir, "mind_descriptors")
        os.makedirs(mind_dir, exist_ok=True)

        # Convert from PyTorch tensors to NumPy arrays and save
        fixed_mind_data = np.transpose(mind_fixed[0].numpy(), (1, 2, 3, 0))
        fixed_mind_nii = nib.Nifti1Image(fixed_mind_data, fixed_nib.affine, fixed_nib.header)
        nib.save(fixed_mind_nii, os.path.join(mind_dir, "fixed_mind.nii.gz"))
        
        # Free memory as we go
        del fixed_mind_data, mind_fixed, fixed_tensor
        gc.collect()

        lamar_mind_data = np.transpose(mind_lamar[0].numpy(), (1, 2, 3, 0))
        lamar_mind_nii = nib.Nifti1Image(lamar_mind_data, lamar_nib.affine, lamar_nib.header)
        nib.save(lamar_mind_nii, os.path.join(mind_dir, "lamar_mind.nii.gz"))
        
        # Free memory
        del lamar_mind_data, mind_lamar, lamar_tensor
        gc.collect()

        ants_mind_data = np.transpose(mind_ants[0].numpy(), (1, 2, 3, 0))
        ants_mind_nii = nib.Nifti1Image(ants_mind_data, ants_nib.affine, ants_nib.header)
        nib.save(ants_mind_nii, os.path.join(mind_dir, "ants_mind.nii.gz"))
        
        # Free memory
        del ants_mind_data, mind_ants, ants_tensor
        gc.collect()
        
        # Save optional mind descriptors
        if mind_lamar_robust is not None:
            lamar_robust_mind_data = np.transpose(mind_lamar_robust[0].numpy(), (1, 2, 3, 0))
            lamar_robust_mind_nii = nib.Nifti1Image(lamar_robust_mind_data, lamar_robust_nib.affine, lamar_robust_nib.header)
            nib.save(lamar_robust_mind_nii, os.path.join(mind_dir, "lamar_robust_mind.nii.gz"))
            del lamar_robust_mind_data, mind_lamar_robust, lamar_robust_tensor
            gc.collect()
            
        if mind_ants_default is not None:
            ants_default_mind_data = np.transpose(mind_ants_default[0].numpy(), (1, 2, 3, 0))
            ants_default_mind_nii = nib.Nifti1Image(ants_default_mind_data, ants_default_nib.affine, ants_default_nib.header)
            nib.save(ants_default_mind_nii, os.path.join(mind_dir, "ants_default_mind.nii.gz"))
            del ants_default_mind_data, mind_ants_default, ants_default_tensor
            gc.collect()

        # Add easyreg data loading
        easyreg_data = None
        easyreg_tensor = None
        easyreg_nib = None
        
        if easyreg_output and os.path.exists(easyreg_output):
            easyreg_nib = nib.load(easyreg_output)
            easyreg_data = easyreg_nib.get_fdata()
            easyreg_tensor = torch.from_numpy(easyreg_data).float().unsqueeze(0).unsqueeze(0)
        
        # Add easyreg MIND calculation
        if easyreg_tensor is not None:
            mind_easyreg = mind_descriptor(easyreg_tensor)
            
        # Save easyreg MIND descriptors
        if mind_easyreg is not None:
            easyreg_mind_data = np.transpose(mind_easyreg[0].numpy(), (1, 2, 3, 0))
            easyreg_mind_nii = nib.Nifti1Image(easyreg_mind_data, easyreg_nib.affine, easyreg_nib.header)
            nib.save(easyreg_mind_nii, os.path.join(mind_dir, "easyreg_mind.nii.gz"))
            del easyreg_mind_data, mind_easyreg, easyreg_tensor
            gc.collect()

        print(f"    MIND descriptors saved in {mind_dir}")
        
        # Final cleanup
        del fixed_data, fixed_nib
        del lamar_data, lamar_nib
        del ants_data, ants_nib
        if lamar_robust_nib is not None:
            del lamar_robust_data, lamar_robust_nib
        if ants_default_nib is not None:
            del ants_default_data, ants_default_nib
        
        # Clear CUDA cache if using GPU
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return mind_dir
    
    except Exception as e:
        print(f"Error generating MIND descriptors: {e}")
        # Ensure memory cleanup even if there's an error
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return None


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
        "fsl_registered.nii.gz",
        "easyreg_registered.nii.gz"  # Add EasyReg output file
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(output_dir, file)):
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch registration between T1w and DWI scans"
    )
    # Add existing arguments
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
    # Add new argument for metrics-only mode
    parser.add_argument(
        "--metrics-only", 
        action="store_true", 
        help="Only calculate metrics on existing registrations (skip registration step)"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define CSV file path
    results_csv = os.path.join(args.output_dir, "registration_results.csv")

    # Define fieldnames for CSV based on the keys in row_data
    fieldnames = [
        "subject_id", "session_id", 
        "lamar_time", "lamar_robust_time", "ants_time", "ants_default_time", 
        "ants_medium_time", "fsl_time", "easyreg_time",
        "speedup_lamar_vs_ants", "speedup_robust_vs_ants", 
        "speedup_lamar_vs_ants_default", "speedup_robust_vs_ants_default",
        "speedup_lamar_vs_ants_medium", "speedup_robust_vs_ants_medium",
        "speedup_lamar_vs_fsl", "speedup_robust_vs_fsl",
        "speedup_lamar_vs_easyreg", "speedup_robust_vs_easyreg",
        "mi_lamar", "mi_lamar_robust", "mi_ants", "mi_ants_default", 
        "mi_ants_medium", "mi_fsl", "mi_easyreg",
        "antsneighborhoodcorrelation_lamar", "antsneighborhoodcorrelation_lamar_robust", 
        "antsneighborhoodcorrelation_ants", "antsneighborhoodcorrelation_ants_default",
        "antsneighborhoodcorrelation_ants_medium", "antsneighborhoodcorrelation_fsl", 
        "antsneighborhoodcorrelation_easyreg",
        "mind_lamar", "mind_lamar_robust", "mind_ants", "mind_ants_default", 
        "mind_ants_medium", "mind_fsl", "mind_easyreg",
        "ngf_lamar", "ngf_lamar_robust", "ngf_ants", "ngf_ants_default", 
        "ngf_ants_medium", "ngf_fsl", "ngf_easyreg",
        "mind_descriptors_dir"
    ]

    # Create CSV file with headers if it doesn't exist
    if not os.path.isfile(results_csv):
        with open(results_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            print(f"Created new results CSV at {results_csv}")

    # Read existing CSV data to determine which subjects/sessions have already been processed
    processed_sessions = set()
    if os.path.isfile(results_csv):
        try:
            with open(results_csv, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    subject_id = row.get('subject_id')
                    session_id = row.get('session_id')
                    if subject_id and session_id:
                        processed_sessions.add(f"{subject_id}_{session_id}")
            print(f"Found {len(processed_sessions)} previously processed sessions in CSV")
        except Exception as e:
            print(f"Warning: Error reading existing CSV file: {e}")

    # At the top level, add a counter to force GC every few subjects
    subject_count = 0

    # Iterate through subjects
    for subject_folder in os.listdir(args.data_path):
        subject_dir = os.path.join(args.data_path, subject_folder)
        if not os.path.isdir(subject_dir) or not subject_folder.startswith("sub-"):
            continue
        
        # Skip subjects that don't have "HC" in their name
        if "HC" not in subject_folder:
            print(f"Skipping non-HC subject: {subject_folder}")
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

                # Skip already processed sessions
                if f"{subject_folder}_{session_folder}" in processed_sessions and not args.force:
                    print(f"    Session {session_folder} already processed, skipping")
                    continue

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
                    fsl_output = os.path.join(subj_output_dir, "fsl_registered.nii.gz")
                    easyreg_output = os.path.join(subj_output_dir, "easyreg_registered.nii.gz")
                    
                    # Initialize registration times
                    lamar_time = 0
                    lamar_robust_time = 0
                    ants_time = 0
                    ants_default_time = 0
                    ants_medium_time = 0
                    fsl_time = 0
                    easyreg_time = 0
                    
                    # Skip registration if metrics-only mode is enabled
                    if not args.metrics_only:
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

                        # Run EasyReg registration
                        print("    Running FreeSurfer EasyReg registration...")
                        easyreg_time, easyreg_output = run_easyreg_registration(
                            moving_img=dwi_file,
                            fixed_img=t1w_file,
                            output_dir=subj_output_dir,
                            threads=args.threads,
                            verbose=not args.quiet,
                            force=args.force,
                        )

                    else:
                        print("    Metrics-only mode: Skipping registration steps")
                        # Check if output files exist
                        existing_files = []
                        if os.path.exists(lamar_output):
                            existing_files.append("LaMAR")
                        if os.path.exists(lamar_robust_output):
                            existing_files.append("LaMAR robust")
                        if os.path.exists(ants_output):
                            existing_files.append("ANTs")
                        if os.path.exists(ants_default_output):
                            existing_files.append("ANTs default")
                        if os.path.exists(ants_medium_output):
                            existing_files.append("ANTs medium")
                        if os.path.exists(fsl_output):
                            existing_files.append("FSL")
                        if os.path.exists(easyreg_output):
                            existing_files.append("EasyReg")
                            
                        if not existing_files:
                            print("    No existing registration outputs found, skipping")
                            continue
                            
                        print(f"    Found existing outputs: {', '.join(existing_files)}")

                    # Check if any outputs exist for evaluation
                    outputs_exist = (
                        os.path.exists(lamar_output) or
                        os.path.exists(lamar_robust_output) or
                        os.path.exists(ants_output) or
                        os.path.exists(ants_default_output) or
                        os.path.exists(ants_medium_output) or
                        os.path.exists(fsl_output) or
                        os.path.exists(easyreg_output)
                    )
                    

                    
                    # Skip quality assessment if no outputs exist
                    if outputs_exist:
                        # First check if metrics already exist in CSV
                        print("    Comparing registration quality...")
                        quality_results = compare_registration_quality(
                            lamar_output=lamar_output if os.path.exists(lamar_output) else None,
                            ants_output=ants_output if os.path.exists(ants_output) else None,
                            fixed_img=t1w_file,
                            lamar_robust_output=lamar_robust_output if os.path.exists(lamar_robust_output) else None,
                            ants_default_output=ants_default_output if os.path.exists(ants_default_output) else None,
                            ants_medium_output=ants_medium_output if os.path.exists(ants_medium_output) else None,
                            fsl_output=fsl_output if os.path.exists(fsl_output) else None,
                            easyreg_output=easyreg_output if os.path.exists(easyreg_output) else None,
                            subject_id=subject_folder,
                            session_id=session_folder,
                            results_csv=results_csv
                        )
                    else:
                        print("    No registration outputs exist, skipping quality assessment")
                        continue

                    # Initialize empty metrics if no quality results
                    if not quality_results:
                        mi = {}
                        antsneighborhoodcorrelation = {}
                        mind = {}
                        ngf = {}
                    else:
                        # Get metrics
                        mi = quality_results.get("mi", {})
                        antsneighborhoodcorrelation = quality_results.get("antsneighborhoodcorrelation", {})
                        mind = quality_results.get("mind", {})
                        ngf = quality_results.get("ngf", {})

                    # Create row data with all methods including FSL
                    row_data = {
                        "subject_id": subject_folder,
                        "session_id": session_folder,
                        "lamar_time": f"{lamar_time:.2f}" if not args.metrics_only else "N/A",
                        "lamar_robust_time": f"{lamar_robust_time:.2f}" if not args.metrics_only else "N/A",
                        "ants_time": f"{ants_time:.2f}" if not args.metrics_only else "N/A",
                        "ants_default_time": f"{ants_default_time:.2f}" if not args.metrics_only else "N/A",
                        "ants_medium_time": f"{ants_medium_time:.2f}" if not args.metrics_only else "N/A",
                        "fsl_time": f"{fsl_time:.2f}" if not args.metrics_only else "N/A",
                        "easyreg_time": f"{easyreg_time:.2f}" if not args.metrics_only else "N/A",
                        
                        # Speedup metrics are N/A in metrics-only mode
                        "speedup_lamar_vs_ants": f"{ants_time/lamar_time:.2f}" if not args.metrics_only and lamar_time > 0 and ants_time > 0 else "N/A",
                        "speedup_robust_vs_ants": f"{ants_time/lamar_robust_time:.2f}" if not args.metrics_only and lamar_robust_time > 0 and ants_time > 0 else "N/A",
                        "speedup_lamar_vs_ants_default": f"{ants_default_time/lamar_time:.2f}" if not args.metrics_only and lamar_time > 0 and ants_default_time > 0 else "N/A",
                        "speedup_robust_vs_ants_default": f"{ants_default_time/lamar_robust_time:.2f}" if not args.metrics_only and lamar_robust_time > 0 and ants_default_time > 0 else "N/A",
                        "speedup_lamar_vs_ants_medium": f"{ants_medium_time/lamar_time:.2f}" if not args.metrics_only and lamar_time > 0 and ants_medium_time > 0 else "N/A",
                        "speedup_robust_vs_ants_medium": f"{ants_medium_time/lamar_robust_time:.2f}" if not args.metrics_only and lamar_robust_time > 0 and ants_medium_time > 0 else "N/A",
                        "speedup_lamar_vs_fsl": f"{fsl_time/lamar_time:.2f}" if not args.metrics_only and lamar_time > 0 and fsl_time > 0 else "N/A",
                        "speedup_robust_vs_fsl": f"{fsl_time/lamar_robust_time:.2f}" if not args.metrics_only and lamar_robust_time > 0 and fsl_time > 0 else "N/A",
                        "speedup_lamar_vs_easyreg": f"{easyreg_time/lamar_time:.2f}" if not args.metrics_only and lamar_time > 0 and easyreg_time > 0 else "N/A",
                        "speedup_robust_vs_easyreg": f"{easyreg_time/lamar_robust_time:.2f}" if not args.metrics_only and lamar_robust_time > 0 and easyreg_time > 0 else "N/A",
                        "mi_lamar": f"{mi.get('lamar', 'N/A')}" if mi else "N/A",
                        "mi_lamar_robust": f"{mi.get('lamar_robust', 'N/A')}" if mi else "N/A",
                        "mi_ants": f"{mi.get('ants', 'N/A')}" if mi else "N/A",
                        "mi_ants_default": f"{mi.get('ants_default', 'N/A')}" if mi else "N/A",
                        "mi_ants_medium": f"{mi.get('ants_medium', 'N/A')}" if mi else "N/A",
                        "mi_fsl": f"{mi.get('fsl', 'N/A')}" if mi else "N/A",
                        "mi_easyreg": f"{mi.get('easyreg', 'N/A')}" if mi else "N/A",  # Add this
                        "antsneighborhoodcorrelation_lamar": f"{antsneighborhoodcorrelation.get('lamar', 'N/A')}" if antsneighborhoodcorrelation else "N/A",
                        "antsneighborhoodcorrelation_lamar_robust": f"{antsneighborhoodcorrelation.get('lamar_robust', 'N/A')}" if antsneighborhoodcorrelation else "N/A",
                        "antsneighborhoodcorrelation_ants": f"{antsneighborhoodcorrelation.get('ants', 'N/A')}" if antsneighborhoodcorrelation else "N/A",
                        "antsneighborhoodcorrelation_ants_default": f"{antsneighborhoodcorrelation.get('ants_default', 'N/A')}" if antsneighborhoodcorrelation else "N/A",
                        "antsneighborhoodcorrelation_ants_medium": f"{antsneighborhoodcorrelation.get('ants_medium', 'N/A')}" if antsneighborhoodcorrelation else "N/A",
                        "antsneighborhoodcorrelation_fsl": f"{antsneighborhoodcorrelation.get('fsl', 'N/A')}" if antsneighborhoodcorrelation else "N/A",
                        "antsneighborhoodcorrelation_easyreg": f"{antsneighborhoodcorrelation.get('easyreg', 'N/A')}" if antsneighborhoodcorrelation else "N/A",  # Add this
                        "mind_lamar": f"{mind.get('lamar', 'N/A')}" if mind else "N/A",
                        "mind_lamar_robust": f"{mind.get('lamar_robust', 'N/A')}" if mind else "N/A",
                        "mind_ants": f"{mind.get('ants', 'N/A')}" if mind else "N/A",
                        "mind_ants_default": f"{mind.get('ants_default', 'N/A')}" if mind else "N/A",
                        "mind_ants_medium": f"{mind.get('ants_medium', 'N/A')}" if mind else "N/A",
                        "mind_fsl": f"{mind.get('fsl', 'N/A')}" if mind else "N/A",
                        "mind_easyreg": f"{mind.get('easyreg', 'N/A')}" if mind else "N/A",  # Add this
                        "ngf_lamar": f"{ngf.get('lamar', 'N/A')}" if ngf else "N/A",
                        "ngf_lamar_robust": f"{ngf.get('lamar_robust', 'N/A')}" if ngf else "N/A",
                        "ngf_ants": f"{ngf.get('ants', 'N/A')}" if ngf else "N/A",
                        "ngf_ants_default": f"{ngf.get('ants_default', 'N/A')}" if ngf else "N/A",
                        "ngf_ants_medium": f"{ngf.get('ants_medium', 'N/A')}" if ngf else "N/A",
                        "ngf_fsl": f"{ngf.get('fsl', 'N/A')}" if ngf else "N/A",
                        "ngf_easyreg": f"{ngf.get('easyreg', 'N/A')}" if ngf else "N/A",  # Add this
                    }

                    # Extract and save MIND descriptors if requested
                    if args.save_mind:
                        mind_dir = save_mind_descriptors(
                            moving_img=dwi_file,
                            fixed_img=t1w_file,
                            lamar_output=lamar_output,
                            ants_output=ants_output,
                            output_dir=subj_output_dir,
                            lamar_robust_output=lamar_robust_output,
                            ants_default_output=ants_default_output,
                            ants_medium_output=ants_medium_output,
                            fsl_output=fsl_output,
                            easyreg_output=easyreg_output  # Add this
                        )
                        row_data["mind_descriptors_dir"] = mind_dir

                    # Write results to CSV by reopening in append mode
                    with open(results_csv, "a", newline="") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow(row_data)

                    # Update print statements
                    print(f"    Completed registration for {subject_folder}_{session_folder}")
                    print(f"    Times: LaMAR: {lamar_time:.2f}s, LaMAR robust: {lamar_robust_time:.2f}s, "
                          f"ANTs: {ants_time:.2f}s, ANTs default: {ants_default_time:.2f}s, "
                          f"ANTs medium: {ants_medium_time:.2f}s, FSL: {fsl_time:.2f}s, "
                          f"EasyReg: {easyreg_time:.2f}s")  # Add EasyReg time
                    
                    print(f"    Quality metrics:")
                    print(f"      MI (LaMAR: {mi.get('lamar', 'N/A')}, LaMAR robust: {mi.get('lamar_robust', 'N/A')}, "
                          f"ANTs: {mi.get('ants', 'N/A')}, ANTs default: {mi.get('ants_default', 'N/A')}, "
                          f"ANTs medium: {mi.get('ants_medium', 'N/A')}, FSL: {mi.get('fsl', 'N/A')}, "
                          f"EasyReg: {mi.get('easyreg', 'N/A')})")  # Add EasyReg MI
                    
                    print(f"      antsneighborhoodcorrelation (LaMAR: {antsneighborhoodcorrelation.get('lamar', 'N/A')}, LaMAR robust: {antsneighborhoodcorrelation.get('lamar_robust', 'N/A')}, ANTs: {antsneighborhoodcorrelation.get('ants', 'N/A')}, ANTs default: {antsneighborhoodcorrelation.get('ants_default', 'N/A')}, ANTs medium: {antsneighborhoodcorrelation.get('ants_medium', 'N/A')}, FSL: {antsneighborhoodcorrelation.get('fsl', 'N/A')}, EasyReg: {antsneighborhoodcorrelation.get('easyreg', 'N/A')})")
                    print(f"      MIND (LaMAR: {mind.get('lamar', 'N/A')}, LaMAR robust: {mind.get('lamar_robust', 'N/A')}, ANTs: {mind.get('ants', 'N/A')}, ANTs default: {mind.get('ants_default', 'N/A')}, ANTs medium: {mind.get('ants_medium', 'N/A')}, FSL: {mind.get('fsl', 'N/A')}, EasyReg: {mind.get('easyreg', 'N/A')})")
                    print(f"      NGF (LaMAR: {ngf.get('lamar', 'N/A')}, LaMAR robust: {ngf.get('lamar_robust', 'N/A')}, ANTs: {ngf.get('ants', 'N/A')}, ANTs default: {ngf.get('ants_default', 'N/A')}, ANTs medium: {ngf.get('ants_medium', 'N/A')}, FSL: {ngf.get('fsl', 'N/A')}, EasyReg: {ngf.get('easyreg', 'N/A')})")
                    
                    print(f"    Completed registration for {subject_folder}_{session_folder}")
                    print(f"    Times: LaMAR: {lamar_time:.2f}s, LaMAR robust: {lamar_robust_time:.2f}s, "
                          f"ANTs: {ants_time:.2f}s, ANTs default: {ants_default_time:.2f}s, "
                          f"ANTs medium: {ants_medium_time:.2f}s, FSL: {fsl_time:.2f}s, "
                          f"EasyReg: {easyreg_time:.2f}s")  # Add EasyReg time

                    # Free up memory
                    if 'quality_results' in locals():
                        del quality_results
                    if 'mi' in locals():
                        del mi
                    if 'antsneighborhoodcorrelation' in locals():
                        del antsneighborhoodcorrelation
                    if 'mind' in locals():
                        del mind
                    if 'ngf' in locals():
                        del ngf
                        
                    # Clear any PyTorch cache
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    # Force garbage collection
                    gc.collect()
                    
                else:
                    print(f"    Missing T1w or DWI scan, skipping")
            except Exception as e:
                print(f"    Error processing session {session_folder}: {e}")
                continue

        # Increment subject counter and perform GC every few subjects
        subject_count += 1
        if subject_count % 3 == 0:  # Every 3 subjects
            print("Performing thorough memory cleanup...")
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\nRegistration batch processing complete. Results saved to {results_csv}")


if __name__ == "__main__":
    main()
