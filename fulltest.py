import os
import csv
import argparse
import numpy as np
import nibabel as nib
import torch
from runtest import run_lamar_registration, run_direct_ants_registration, compare_registration_quality
from torch_mind import MIND3D

def save_mind_descriptors(moving_img, fixed_img, lamar_output, ants_output, output_dir):
    """Extract and save MIND descriptors as multi-channel NIfTI images."""
    print("    Generating MIND descriptor visualizations...")
    
    # Load images using nibabel
    fixed_nib = nib.load(fixed_img)
    fixed_data = fixed_nib.get_fdata()
    lamar_nib = nib.load(lamar_output)
    lamar_data = lamar_nib.get_fdata()
    ants_nib = nib.load(ants_output)
    ants_data = ants_nib.get_fdata()
    
    # Convert to tensors with batch dimension for MIND
    fixed_tensor = torch.from_numpy(fixed_data).float().unsqueeze(0).unsqueeze(0)
    lamar_tensor = torch.from_numpy(lamar_data).float().unsqueeze(0).unsqueeze(0)
    ants_tensor = torch.from_numpy(ants_data).float().unsqueeze(0).unsqueeze(0)
    
    # Create MIND descriptor
    mind_descriptor = MIND3D(patch_size=3, sigma=0.5)
    
    # Calculate MIND descriptors
    with torch.no_grad():
        mind_fixed = mind_descriptor(fixed_tensor)
        mind_lamar = mind_descriptor(lamar_tensor)
        mind_ants = mind_descriptor(ants_tensor)
    
    # Create output directory for MIND visualizations
    mind_dir = os.path.join(output_dir, "mind_descriptors")
    os.makedirs(mind_dir, exist_ok=True)
    
    # Convert from PyTorch tensors to NumPy arrays
    # Move channels dimension to the end (NIfTI convention)
    # From [B,C,D,H,W] to [D,H,W,C]
    fixed_mind_data = np.transpose(mind_fixed[0].numpy(), (1, 2, 3, 0))
    lamar_mind_data = np.transpose(mind_lamar[0].numpy(), (1, 2, 3, 0))
    ants_mind_data = np.transpose(mind_ants[0].numpy(), (1, 2, 3, 0))
    
    # Save multi-channel MIND descriptors as NIfTI files
    fixed_mind_nii = nib.Nifti1Image(fixed_mind_data, fixed_nib.affine, fixed_nib.header)
    nib.save(fixed_mind_nii, os.path.join(mind_dir, "fixed_mind.nii.gz"))
    
    lamar_mind_nii = nib.Nifti1Image(lamar_mind_data, lamar_nib.affine, lamar_nib.header)
    nib.save(lamar_mind_nii, os.path.join(mind_dir, "lamar_mind.nii.gz"))
    
    ants_mind_nii = nib.Nifti1Image(ants_mind_data, ants_nib.affine, ants_nib.header)
    nib.save(ants_mind_nii, os.path.join(mind_dir, "ants_mind.nii.gz"))
    
    print(f"    MIND descriptors saved in {mind_dir}")
    return mind_dir

def normalize_for_vis(array):
    """Normalize array to 0-1 range for visualization."""
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val + 1e-8)

def main():
    parser = argparse.ArgumentParser(description="Batch registration between T1w and DWI scans")
    parser.add_argument("--data-path", default="/data/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0",
                        help="Path to BIDS dataset")
    parser.add_argument("--output-dir", default="./registration_results",
                        help="Directory for output files and results")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads to use")
    parser.add_argument("--registration-method", default="SyNRA", help="Registration method")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--save-mind", action="store_true", help="Save MIND descriptor visualizations")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define CSV file path
    results_csv = os.path.join(args.output_dir, "registration_results.csv")
    fieldnames = ['subject_id', 'session_id', 'lamar_time', 'ants_time', 'speedup', 
                  'nmi_lamar', 'nmi_ants', 'mind_lamar', 'mind_ants', 'ngf_lamar', 'ngf_ants',
                  'mind_descriptors_dir']
    
    # Create CSV file with headers if it doesn't exist
    file_exists = os.path.isfile(results_csv)
    if not file_exists:
        with open(results_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    # Iterate through subjects
    for subject_folder in os.listdir(args.data_path):
        subject_dir = os.path.join(args.data_path, subject_folder)
        if not os.path.isdir(subject_dir) or not subject_folder.startswith('sub-'):
            continue
            
        print(f"Processing subject: {subject_folder}")
        
        # Iterate through sessions
        for session_folder in os.listdir(subject_dir):
            try:
                session_dir = os.path.join(subject_dir, session_folder)
                if not os.path.isdir(session_dir) or not session_folder.startswith('ses-'):
                    continue
                    
                print(f"  Processing session: {session_folder}")
                
                # Find T1w and DWI scans
                t1w_file = None
                dwi_file = None
                
                # Check for T1w scan
                anatomical_folder = os.path.join(subject_dir, session_folder, "anat")
                if os.path.isdir(anatomical_folder):
                    t1w_path = os.path.join(anatomical_folder, 
                                        f"{subject_folder}_{session_folder}_space-nativepro_T1w.nii.gz")
                    if os.path.isfile(t1w_path):
                        t1w_file = t1w_path
                
                # Check for DWI scan
                diffusion_folder = os.path.join(subject_dir, session_folder, "dwi")
                if os.path.isdir(diffusion_folder):
                    dwi_path = os.path.join(diffusion_folder,
                                        f"{subject_folder}_{session_folder}_space-dwi_desc-b0.nii.gz")
                    if os.path.isfile(dwi_path):
                        dwi_file = dwi_path
                
                # Only proceed if both scans exist
                if t1w_file and dwi_file:
                    print(f"    Found both T1w and DWI scans, performing registration")
                    
                    # Create subject-specific output directory
                    subj_output_dir = os.path.join(args.output_dir, f"{subject_folder}_{session_folder}")
                    os.makedirs(subj_output_dir, exist_ok=True)
                    
                    # Run LaMAR registration (DWI to T1w)
                    print("    Running LaMAR registration...")
                    lamar_time, lamar_output = run_lamar_registration(
                        moving_img=dwi_file,
                        fixed_img=t1w_file,
                        output_dir=subj_output_dir,
                        registration_method=args.registration_method,
                        threads=args.threads,
                        verbose=not args.quiet
                    )
                    
                    # Run Direct ANTs registration
                    print("    Running ANTs registration...")
                    ants_time, ants_output = run_direct_ants_registration(
                        moving_img=dwi_file,
                        fixed_img=t1w_file,
                        output_dir=subj_output_dir,
                        registration_method=args.registration_method,
                        threads=args.threads,
                        verbose=not args.quiet
                    )
                    
                    # Compare registration quality
                    print("    Comparing registration quality...")
                    quality_results = compare_registration_quality(
                        lamar_output=lamar_output,
                        ants_output=ants_output,
                        fixed_img=t1w_file
                    )
                    
                    # Calculate speedup
                    speedup = ants_time / lamar_time if lamar_time > 0 else 0
                    
                    # Get metrics (handle cases where metrics might not be available)
                    nmi = quality_results.get('nmi', {})
                    mind = quality_results.get('mind', {})
                    ngf = quality_results.get('ngf', {})
                    
                    # Create row data
                    row_data = {
                        'subject_id': subject_folder,
                        'session_id': session_folder,
                        'lamar_time': f"{lamar_time:.2f}",
                        'ants_time': f"{ants_time:.2f}",
                        'speedup': f"{speedup:.2f}",
                        'nmi_lamar': f"{nmi.get('lamar', 'N/A')}" if nmi else 'N/A',
                        'nmi_ants': f"{nmi.get('ants', 'N/A')}" if nmi else 'N/A',
                        'mind_lamar': f"{mind.get('lamar', 'N/A')}" if mind else 'N/A',
                        'mind_ants': f"{mind.get('ants', 'N/A')}" if mind else 'N/A',
                        'ngf_lamar': f"{ngf.get('lamar', 'N/A')}" if ngf else 'N/A',
                        'ngf_ants': f"{ngf.get('ants', 'N/A')}" if ngf else 'N/A',
                    }
                    
                    # Extract and save MIND descriptors if requested
                    if args.save_mind:
                        mind_dir = save_mind_descriptors(
                            moving_img=dwi_file,
                            fixed_img=t1w_file,
                            lamar_output=lamar_output,
                            ants_output=ants_output,
                            output_dir=subj_output_dir
                        )
                        row_data['mind_descriptors_dir'] = mind_dir
                    
                    # Write results to CSV by reopening in append mode
                    with open(results_csv, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow(row_data)
                        
                    print(f"    Quality metrics: NMI (LaMAR: {nmi.get('lamar', 'N/A')}, ANTs: {nmi.get('ants', 'N/A')})")
                    print(f"    MIND (LaMAR: {mind.get('lamar', 'N/A')}, ANTs: {mind.get('ants', 'N/A')})")
                    print(f"    NGF (LaMAR: {ngf.get('lamar', 'N/A')}, ANTs: {ngf.get('ants', 'N/A')})")
                    print(f"    Completed registration for {subject_folder}_{session_folder}")
                    print(f"    LaMAR time: {lamar_time:.2f}s, ANTs time: {ants_time:.2f}s, Speedup: {speedup:.2f}x")
                    print(f"    Results appended to {results_csv}")
                else:
                    print(f"    Missing T1w or DWI scan, skipping")
            except Exception as e:
                print(f"    Error processing session {session_folder}: {e}")
                continue
    
    print(f"\nRegistration batch processing complete. Results saved to {results_csv}")

if __name__ == "__main__":
    main()
