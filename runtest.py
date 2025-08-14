#!/usr/bin/env python3
"""
Benchmark test comparing registration speed between LaMAR and direct ANTs registration.

This script runs registration on the same image pair using both:
1. LaMAR's parcellation-based approach (via SynthSeg)
2. Direct ANTs registration using the same parameters

It measures and reports the execution time for both methods.
"""

import os
import sys
import time
import shutil
import tempfile
import argparse
import subprocess
import nibabel as nib
import numpy as np
import torch
import ants
import csv

def run_lamar_registration(
    moving_img,
    fixed_img,
    output_dir,
    registration_method="SyNRA",
    threads=1,
    verbose=True,
    force=False,
):
    """Run registration using LaMAR CLI and measure time."""
    # Set up output paths
    output_img = os.path.join(output_dir, "lamar_registered.nii.gz")
    
    # Skip if output already exists and not forced to rerun
    if os.path.exists(output_img) and not force:
        if verbose:
            print(f"Output {output_img} already exists. Skipping LaMAR registration.")
        return 0.0, output_img
    
    start_time = time.time()

    # Setup paths for intermediate files
    moving_parc = os.path.join(output_dir, "moving_parc.nii.gz")
    fixed_parc = os.path.join(output_dir, "fixed_parc.nii.gz")
    registered_parc = os.path.join(output_dir, "registered_parc.nii.gz")
    affine_file = os.path.join(output_dir, "lamar_affine.mat")
    warp_file = os.path.join(output_dir, "lamar_warp.nii.gz")

    # Build command for lamar registration
    cmd = [
        "lamar",
        "register",
        "--moving",
        moving_img,
        "--fixed",
        fixed_img,
        "--output",
        output_img,
        "--moving-parc",
        moving_parc,
        "--fixed-parc",
        fixed_parc,
        "--registered-parc",
        registered_parc,
        "--affine",
        affine_file,
        "--warpfield",
        warp_file,
        "--registration-method",
        registration_method,
        "--synthseg-threads",
        str(threads),
        "--ants-threads",
        str(threads),
        "--skip-qc",
    ]

    # Run LaMAR registration
    if verbose:
        subprocess.run(cmd, check=True)
    else:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    elapsed_time = time.time() - start_time
    return elapsed_time, output_img


def run_lamar_registration_robust(
    moving_img,
    fixed_img,
    output_dir,
    registration_method="SyNRA",
    threads=1,
    verbose=True,
    force=False,
):
    """Run registration using LaMAR coregister with initial transform and measure time."""
    # Set up output paths
    output_img = os.path.join(output_dir, "lamar_robust_registered.nii.gz")
    
    # Skip if output already exists and not forced to rerun
    if os.path.exists(output_img) and not force:
        if verbose:
            print(f"Output {output_img} already exists. Skipping LaMAR robust registration.")
        return 0.0, output_img
    
    start_time = time.time()

    # Setup paths for intermediate files
    registered_parc = os.path.join(output_dir, "registered_parc_robust.nii.gz")
    affine_file = os.path.join(output_dir, "lamar_robust_affine.mat")
    warp_file = os.path.join(output_dir, "lamar_robust_warp.nii.gz")
    
    # Initial transform files from standard registration
    initial_affine_file = os.path.join(output_dir, "lamar_affine.mat")
    initial_warp_file = os.path.join(output_dir, "lamar_warp.nii.gz")

    # Build command for lamar coregister with initial transforms
    cmd = [
        "lamar", "coregister",
        "--fixed", fixed_img,
        "--moving", moving_img,
        "--output", output_img,
        "--warp-file", warp_file,
        "--affine-file", affine_file,
        "--registration-method", "SyNOnly",
        "--initial-affine-file", initial_affine_file,
        "--initial-warp-file", initial_warp_file,
        "--interpolator", "linear",
        "--reg-iterations","10, 20",
    ]
    
    # Add additional parameters
    if verbose:
        cmd.append("--verbose")
    
    # Add thread parameters if coregister supports them
    if threads > 1:
        env = os.environ.copy()
        env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
        env["OMP_NUM_THREADS"] = str(threads)

    # Run LaMAR coregister
    if verbose:
        subprocess.run(cmd, check=True, env=env if threads > 1 else None)
    else:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                      env=env if threads > 1 else None)

    elapsed_time = time.time() - start_time
    return elapsed_time, output_img


def run_direct_ants_registration(
    moving_img,
    fixed_img,
    output_dir,
    registration_method="SyNRA",
    threads=1,
    verbose=True,
    force=False,
):
    """Run direct ANTs registration via ANTsPyX and measure time."""
    # Set up output paths
    output_img = os.path.join(output_dir, "direct_ants_registered.nii.gz")
    
    # Skip if output already exists and not forced to rerun
    if os.path.exists(output_img) and not force:
        if verbose:
            print(f"Output {output_img} already exists. Skipping ANTs registration.")
        return 0.0, output_img
    
    import ants
    start_time = time.time()

    env = os.environ.copy()
    # Set ANTs/ITK thread count
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)  # OpenMP threads for ANTs

    # Load images
    fixed_image = ants.image_read(fixed_img)
    moving_image = ants.image_read(moving_img)

    # Map registration method to ANTs type
    type_of_transform = registration_method

    # Log if verbose
    if verbose:
        print(f"Running ANTsPyX registration with method: {type_of_transform}")
        print(f"Thread count: {threads}")
        print(f"Moving image: {moving_img}")
        print(f"Fixed image: {fixed_img}")

        # Perform registration
    registration = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform=type_of_transform,
        verbose=verbose,
        reg_iterations=(10,20)
    )

    # Save outputs
    ants.image_write(registration["warpedmovout"], output_img)

    temp_files_to_delete = set(registration['fwdtransforms'] + registration['invtransforms'])
    deleted_count = 0
    for temp_file in temp_files_to_delete:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                deleted_count += 1
        except OSError as e:
            print(f"Warning: Could not remove temporary file {temp_file}: {e}")
    print(f"Successfully cleaned up {deleted_count} temporary files.")

    elapsed_time = time.time() - start_time
    return elapsed_time, output_img


def run_direct_ants_registration_default(
    moving_img,
    fixed_img,
    output_dir,
    registration_method="SyNRA",
    threads=1,
    verbose=True,
    force=False,
):
    """Run direct ANTs registration via ANTsPyX with default parameters and measure time."""
    # Set up output paths
    output_img = os.path.join(output_dir, "direct_ants_default_registered.nii.gz")
    
    # Skip if output already exists and not forced to rerun
    if os.path.exists(output_img) and not force:
        if verbose:
            print(f"Output {output_img} already exists. Skipping ANTs default registration.")
        return 0.0, output_img
    
    import ants
    start_time = time.time()

    env = os.environ.copy()
    # Set ANTs/ITK thread count
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)  # OpenMP threads for ANTs

    # Load images
    fixed_image = ants.image_read(fixed_img)
    moving_image = ants.image_read(moving_img)

    # Map registration method to ANTs type
    type_of_transform = registration_method

    # Log if verbose
    if verbose:
        print(f"Running ANTsPyX registration with default parameters, method: {type_of_transform}")
        print(f"Thread count: {threads}")
        print(f"Moving image: {moving_img}")
        print(f"Fixed image: {fixed_img}")

    # Perform registration with default parameters
    registration = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform=type_of_transform,
        verbose=verbose,
        # No reg_iterations parameter means using ANTs defaults
    )

    # Save outputs
    ants.image_write(registration["warpedmovout"], output_img)

    temp_files_to_delete = set(registration['fwdtransforms'] + registration['invtransforms'])
    deleted_count = 0
    for temp_file in temp_files_to_delete:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                deleted_count += 1
        except OSError as e:
            print(f"Warning: Could not remove temporary file {temp_file}: {e}")
    print(f"Successfully cleaned up {deleted_count} temporary files.")

    elapsed_time = time.time() - start_time
    return elapsed_time, output_img


def run_direct_ants_registration_medium_iters(
    moving_img,
    fixed_img,
    output_dir,
    registration_method="SyNRA",
    threads=1,
    verbose=True,
    force=False,
):
    """Run direct ANTs registration via ANTsPyX with medium iterations (40,20,10) and measure time."""
    # Set up output paths
    output_img = os.path.join(output_dir, "direct_ants_medium_registered.nii.gz")
    
    # Skip if output already exists and not forced to rerun
    if os.path.exists(output_img) and not force:
        if verbose:
            print(f"Output {output_img} already exists. Skipping ANTs medium registration.")
        return 0.0, output_img
    
    import ants
    start_time = time.time()

    env = os.environ.copy()
    # Set ANTs/ITK thread count
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)  # OpenMP threads for ANTs

    # Load images
    fixed_image = ants.image_read(fixed_img)
    moving_image = ants.image_read(moving_img)

    # Map registration method to ANTs type
    type_of_transform = registration_method

    # Log if verbose
    if verbose:
        print(f"Running ANTsPyX registration with medium iterations (40,20,10), method: {type_of_transform}")
        print(f"Thread count: {threads}")
        print(f"Moving image: {moving_img}")
        print(f"Fixed image: {fixed_img}")

    # Perform registration with medium iterations
    registration = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform=type_of_transform,
        verbose=verbose,
        reg_iterations=(40, 20, 10)  # Medium level of iterations
    )

    # Save outputs
    ants.image_write(registration["warpedmovout"], output_img)

    temp_files_to_delete = set(registration['fwdtransforms'] + registration['invtransforms'])
    deleted_count = 0
    for temp_file in temp_files_to_delete:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                deleted_count += 1
        except OSError as e:
            print(f"Warning: Could not remove temporary file {temp_file}: {e}")
    print(f"Successfully cleaned up {deleted_count} temporary files.")

    elapsed_time = time.time() - start_time
    return elapsed_time, output_img


def run_fsl_registration(
    moving_img,
    fixed_img,
    output_dir,
    threads=1,
    verbose=True,
    force=False,
):
    """Run registration using FSL's epi_reg (BBR) and measure time."""
    # Import here to avoid requiring FSL for other functions
    from fsl.wrappers import epi_reg, fslmaths, bet, flirt
    
    # Set up output paths
    output_img = os.path.join(output_dir, "fsl_registered.nii.gz")
    
    # Skip if output already exists and not forced to rerun
    if os.path.exists(output_img) and not force:
        if verbose:
            print(f"Output {output_img} already exists. Skipping FSL registration.")
        return 0.0, output_img
    
    start_time = time.time()
    
    # Set up paths for intermediate files
    t1_brain = os.path.join(output_dir, "t1_brain.nii.gz")
    epi_reg_prefix = os.path.join(output_dir, "fsl_epi_reg")
    
    try:
        # FSL requires brain-extracted T1 for epi_reg
        if not os.path.exists(t1_brain):
            if verbose:
                print("Brain-extracting T1 image...")
            bet(fixed_img, t1_brain, f=0.5)
        
        # Set FSLPARALLEL if using multiple threads
        if threads > 1:
            os.environ["FSLPARALLEL"] = str(threads)
            os.environ['OMP_NUM_THREADS'] = str(threads)
            os.environ['FSLNUMTHREADS'] = str(threads)
        
        # Run epi_reg to perform EPI to T1 registration using BBR
        if verbose:
            print(f"Running FSL epi_reg with BBR...")
        
        # Run epi_reg to generate transformation
        epi_reg(
            epi=moving_img,
            t1=fixed_img,
            t1brain=t1_brain,
            out=epi_reg_prefix
        )
        
        # Apply the transformation
        affine_mat = f"{epi_reg_prefix}.mat"
        
        if verbose:
            print(f"Applying FSL transformation...")
        
        flirt(
            moving_img,
            fixed_img,
            out=output_img,
            init=affine_mat,
            applyxfm=True
        )
        
    except Exception as e:
        print(f"Error during FSL registration: {e}")
        if os.path.exists(output_img):
            os.remove(output_img)
        elapsed_time = time.time() - start_time
        return elapsed_time, None
    
    elapsed_time = time.time() - start_time
    return elapsed_time, output_img


def run_easyreg_registration(
    moving_img,
    fixed_img,
    output_dir,
    threads=1,
    verbose=True,
    force=False,
    affine_only=False
):
    """Run registration using FreeSurfer's mri_easyreg and measure time."""
    # Set up output paths
    output_img = os.path.join(output_dir, "easyreg_registered.nii.gz")
    
    # Skip if output already exists and not forced to rerun
    if os.path.exists(output_img) and not force:
        if verbose:
            print(f"Output {output_img} already exists. Skipping EasyReg registration.")
        return 0.0, output_img
    
    start_time = time.time()
    
    # Setup paths for intermediate files
    prefix = os.path.join(output_dir, "easyreg")
    ref_seg = f"{prefix}_ref_seg.mgz"
    flo_seg = f"{prefix}_flo_seg.mgz"
    fwd_field = f"{prefix}_fwd_field.mgz"
    bak_field = f"{prefix}_bak_field.mgz"
    
    # Build the mri_easyreg command
    cmd = [
        "python", 
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "easyreg.py"),
        "--ref", fixed_img,
        "--flo", moving_img,
        "--out", prefix,
        "--threads", str(threads)
    ]
    
    if affine_only:
        cmd.append("--affine_only")
        
    if verbose:
        cmd.append("--verbose")
    
    # Run EasyReg registration
    try:
        if verbose:
            print(f"Running EasyReg with command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # EasyReg outputs the registered image as flo_in_ref.mgz
        registered_img = f"{prefix}_flo_in_ref.nii.gz"
        shutil.copy(registered_img, output_img)
            
    except Exception as e:
        print(f"Error during EasyReg registration: {e}")
        if os.path.exists(output_img):
            os.remove(output_img)
        elapsed_time = time.time() - start_time
        return elapsed_time, None
    
    elapsed_time = time.time() - start_time
    return elapsed_time, output_img


def compare_registration_quality(lamar_output, ants_output, fixed_img, lamar_robust_output=None, 
                               ants_default_output=None, ants_medium_output=None, fsl_output=None,
                               easyreg_output=None, subject_id=None, session_id=None, results_csv=None):
    """Compare the registration quality using all available metrics.

    Args:
        lamar_output: Path to LaMAR registered image
        ants_output: Path to ANTs registered image
        fixed_img: Path to fixed reference image
        lamar_robust_output: Path to LaMAR registered image with robust flag (optional)
        ants_default_output: Path to ANTs registered image with default parameters (optional)
        ants_medium_output: Path to ANTs registered image with medium iterations (optional)
        fsl_output: Path to FSL registered image (optional)
        easyreg_output: Path to FreeSurfer EasyReg registered image (optional)
        subject_id: Subject ID for CSV lookup (optional)
        session_id: Session ID for CSV lookup (optional) 
        results_csv: Path to results CSV file (optional)

    Returns:
        Dictionary with results for all metrics or None if no registrations to compare
    """
    # Check if metrics already exist for this subject/session
    if subject_id and session_id and results_csv and os.path.exists(results_csv):
        try:
            with open(results_csv, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row.get('subject_id') == subject_id and row.get('session_id') == session_id:
                        # Check if metrics exist for any method
                        metric_fields = ['mi_', 'antsneighborhoodcorrelation_', 'mind_', 'ngf_']
                        if any(field in key and row[key] != 'N/A' for key in row for field in metric_fields):
                            print(f"    Metrics already exist in CSV for {subject_id}/{session_id}, skipping calculation")
                            return None
        except Exception as e:
            print(f"    Warning: Error checking CSV for existing metrics: {e}")
    
    # First, check if any registration was actually performed
    # We need at least one registration output to compare
    if not os.path.exists(lamar_output) and not os.path.exists(ants_output) and \
       (not lamar_robust_output or not os.path.exists(lamar_robust_output)) and \
       (not ants_default_output or not os.path.exists(ants_default_output)) and \
       (not ants_medium_output or not os.path.exists(ants_medium_output)) and \
       (not fsl_output or not os.path.exists(fsl_output)) and \
       (not easyreg_output or not os.path.exists(easyreg_output)):
        print("No registration outputs available for quality assessment")
        return None
    
    # We need at least two registered images to compare
    valid_outputs = 0
    for img_path in [lamar_output, ants_output, lamar_robust_output, 
                     ants_default_output, ants_medium_output, fsl_output,
                     easyreg_output]:
        if img_path and os.path.exists(img_path):
            valid_outputs += 1
    
    if valid_outputs < 2:
        print("Not enough registration outputs available for quality assessment")
        return None
        
    # Continue with quality assessment as normal...
    results = {}
    
    # Load images that exist
    fixed_img_nib = nib.load(fixed_img)
    fixed_img_data = fixed_img_nib.get_fdata()
    fixed_tensor = torch.from_numpy(fixed_img_data).float().unsqueeze(0).unsqueeze(0)
    
    # Initialize data containers
    lamar_img_data = None
    lamar_tensor = None
    ants_img_data = None
    ants_tensor = None
    lamar_robust_data = None
    lamar_robust_tensor = None
    ants_default_data = None
    ants_default_tensor = None
    ants_medium_data = None
    ants_medium_tensor = None
    fsl_data = None
    fsl_tensor = None
    easyreg_data = None
    easyreg_tensor = None
    
    # Load only available images
    if os.path.exists(lamar_output):
        lamar_img_nib = nib.load(lamar_output)
        lamar_img_data = lamar_img_nib.get_fdata()
        lamar_tensor = torch.from_numpy(lamar_img_data).float().unsqueeze(0).unsqueeze(0)
    
    if os.path.exists(ants_output):
        ants_img_nib = nib.load(ants_output)
        ants_img_data = ants_img_nib.get_fdata()
        ants_tensor = torch.from_numpy(ants_img_data).float().unsqueeze(0).unsqueeze(0)
    
    if lamar_robust_output and os.path.exists(lamar_robust_output):
        lamar_robust_nib = nib.load(lamar_robust_output)
        lamar_robust_data = lamar_robust_nib.get_fdata()
        lamar_robust_tensor = torch.from_numpy(lamar_robust_data).float().unsqueeze(0).unsqueeze(0)
        
    if ants_default_output and os.path.exists(ants_default_output):
        ants_default_nib = nib.load(ants_default_output)
        ants_default_data = ants_default_nib.get_fdata()
        ants_default_tensor = torch.from_numpy(ants_default_data).float().unsqueeze(0).unsqueeze(0)
        
    if ants_medium_output and os.path.exists(ants_medium_output):
        ants_medium_nib = nib.load(ants_medium_output)
        ants_medium_data = ants_medium_nib.get_fdata()
        ants_medium_tensor = torch.from_numpy(ants_medium_data).float().unsqueeze(0).unsqueeze(0)
        
    if fsl_output and os.path.exists(fsl_output):
        fsl_nib = nib.load(fsl_output)
        fsl_data = fsl_nib.get_fdata()
        fsl_tensor = torch.from_numpy(fsl_data).float().unsqueeze(0).unsqueeze(0)

    # Load EasyReg output if available
    if easyreg_output and os.path.exists(easyreg_output):
        easyreg_nib = nib.load(easyreg_output)
        easyreg_data = easyreg_nib.get_fdata()
        easyreg_tensor = torch.from_numpy(easyreg_data).float().unsqueeze(0).unsqueeze(0)

    # Calculate Mutual Information using ANTsPy
    def mutual_information(img1, img2, bins=32):
        """Calculate mutual information between two images using ANTsPy."""
        if img1 is None or img2 is None:
            return None
        # Convert numpy arrays to ANTs images
        img1_ants = ants.from_numpy(img1.astype(np.float32))
        img2_ants = ants.from_numpy(img2.astype(np.float32))
        
        # Calculate mutual information directly with ANTsPy
        return ants.image_mutual_information(img1_ants, img2_ants)

    # Calculate ANTSNeighborhoodCorrelation using ANTsPy
    def ants_neighborhood_correlation(img1, img2):
        """Calculate ANTSNeighborhoodCorrelation between two images using ANTsPy."""
        if img1 is None or img2 is None:
            return None
        # Convert numpy arrays to ANTs images
        img1_ants = ants.from_numpy(img1.astype(np.float32))
        img2_ants = ants.from_numpy(img2.astype(np.float32))
        
        # Use ANTSNeighborhoodCorrelation metric directly
        similarity = ants.image_similarity(img1_ants, img2_ants, metric_type='ANTSNeighborhoodCorrelation')
        
        return similarity

    # Calculate and store MI
    results["mi"] = {}
    if lamar_img_data is not None:
        results["mi"]["lamar"] = mutual_information(lamar_img_data, fixed_img_data)
    if ants_img_data is not None:
        results["mi"]["ants"] = mutual_information(ants_img_data, fixed_img_data)
    
    # Calculate and store ANTSNeighborhoodCorrelation
    results["antsneighborhoodcorrelation"] = {}
    if lamar_img_data is not None:
        results["antsneighborhoodcorrelation"]["lamar"] = ants_neighborhood_correlation(lamar_img_data, fixed_img_data)
    if ants_img_data is not None:
        results["antsneighborhoodcorrelation"]["ants"] = ants_neighborhood_correlation(ants_img_data, fixed_img_data)
    
    # Add robust, default, medium ANTs and FSL results if available
    if lamar_robust_data is not None:
        results["mi"]["lamar_robust"] = mutual_information(lamar_robust_data, fixed_img_data)
        results["antsneighborhoodcorrelation"]["lamar_robust"] = ants_neighborhood_correlation(lamar_robust_data, fixed_img_data)
        
    if ants_default_data is not None:
        results["mi"]["ants_default"] = mutual_information(ants_default_data, fixed_img_data)
        results["antsneighborhoodcorrelation"]["ants_default"] = ants_neighborhood_correlation(ants_default_data, fixed_img_data)
        
    if ants_medium_data is not None:
        results["mi"]["ants_medium"] = mutual_information(ants_medium_data, fixed_img_data)
        results["antsneighborhoodcorrelation"]["ants_medium"] = ants_neighborhood_correlation(ants_medium_data, fixed_img_data)
        
    if fsl_data is not None:
        results["mi"]["fsl"] = mutual_information(fsl_data, fixed_img_data)
        results["antsneighborhoodcorrelation"]["fsl"] = ants_neighborhood_correlation(fsl_data, fixed_img_data)
    # After MI calculations for other methods, add:
    if easyreg_data is not None:
        results["mi"]["easyreg"] = mutual_information(easyreg_data, fixed_img_data)

    # After ANTSNeighborhoodCorrelation calculations for other methods, add:
    if easyreg_data is not None:
        results["antsneighborhoodcorrelation"]["easyreg"] = ants_neighborhood_correlation(easyreg_data, fixed_img_data)
    # Try MIND metric
    try:
        from torch_mind import MINDLoss3D

        mind_loss = MINDLoss3D()
        results["mind"] = {}

        with torch.no_grad():
            if lamar_tensor is not None:
                results["mind"]["lamar"] = mind_loss(lamar_tensor, fixed_tensor).item()
            if ants_tensor is not None:
                results["mind"]["ants"] = mind_loss(ants_tensor, fixed_tensor).item()
            if lamar_robust_tensor is not None:
                results["mind"]["lamar_robust"] = mind_loss(lamar_robust_tensor, fixed_tensor).item()
            if ants_default_tensor is not None:
                results["mind"]["ants_default"] = mind_loss(ants_default_tensor, fixed_tensor).item()
            if ants_medium_tensor is not None:
                results["mind"]["ants_medium"] = mind_loss(ants_medium_tensor, fixed_tensor).item()
            if fsl_tensor is not None:
                results["mind"]["fsl"] = mind_loss(fsl_tensor, fixed_tensor).item()
            if easyreg_tensor is not None:
                results["mind"]["easyreg"] = mind_loss(easyreg_tensor, fixed_tensor).item()
    except Exception as e:
        print(f"Error calculating MIND: {e}")
        results["mind"] = None

    # Try NGF metric
    try:
        from normalized_gradient_field import NormalizedGradientField3d

        pixel_spacing = fixed_img_nib.header.get_zooms()[:3]

        ngf = NormalizedGradientField3d(
            grad_method="default",
            mm_spacing=pixel_spacing,
            reduction="mean",
        )

        results["ngf"] = {}
        
        with torch.no_grad():
            if lamar_tensor is not None:
                results["ngf"]["lamar"] = ngf(lamar_tensor, fixed_tensor).item()
            if ants_tensor is not None:
                results["ngf"]["ants"] = ngf(ants_tensor, fixed_tensor).item()
            if lamar_robust_tensor is not None:
                results["ngf"]["lamar_robust"] = ngf(lamar_robust_tensor, fixed_tensor).item()
            if ants_default_tensor is not None:
                results["ngf"]["ants_default"] = ngf(ants_default_tensor, fixed_tensor).item()
            if ants_medium_tensor is not None:
                results["ngf"]["ants_medium"] = ngf(ants_medium_tensor, fixed_tensor).item()
            if fsl_tensor is not None:
                results["ngf"]["fsl"] = ngf(fsl_tensor, fixed_tensor).item()
            if easyreg_tensor is not None:
                results["ngf"]["easyreg"] = ngf(easyreg_tensor, fixed_tensor).item()
    except Exception as e:
        print(f"Error calculating NGF: {e}")
        results["ngf"] = None

    return results


def main():
    """Run benchmark comparison between LaMAR and direct ANTs registration."""
    parser = argparse.ArgumentParser(
        description="Batch registration between T1w and DWI scans"
    )
    parser.add_argument(
        "--moving", required=True, help="Input moving image to be registered"
    )
    parser.add_argument(
        "--fixed", required=True, help="Reference fixed image (target space)"
    )
    parser.add_argument(
        "--output-dir", default="benchmark_results", help="Directory for output files"
    )
    parser.add_argument(
        "--registration-method", default="SyNRA", help="Registration method"
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="Number of threads to use"
    )
    parser.add_argument(
        "--keep-files", action="store_true", help="Keep intermediate and output files"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress registration output"
    )

    args = parser.parse_args()

    # Validate input files
    if not os.path.isfile(args.moving):
        print(f"Error: Moving image not found: {args.moving}")
        return 1
    if not os.path.isfile(args.fixed):
        print(f"Error: Fixed image not found: {args.fixed}")
        return 1

    # Create output directory or use temporary directory
    temp_dir = None
    if args.output_dir == "./benchmark_results":
        temp_dir = tempfile.mkdtemp(prefix="lamar_benchmark_")
        output_dir = temp_dir
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define CSV file path
    results_csv = os.path.join(args.output_dir, "registration_results.csv")
    
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
    
    # Create CSV file with headers if it doesn't exist
    fieldnames = [
        # Existing fieldnames...
    ]
    
    file_exists = os.path.isfile(results_csv)
    if not file_exists:
        with open(results_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    try:
        print(f"Running benchmark with {args.threads} thread(s)...")
        print(f"Registration method: {args.registration_method}")

        # Run LaMAR registration
        print("\n--- Running LaMAR registration ---")
        lamar_time, lamar_output = run_lamar_registration(
            args.moving,
            args.fixed,
            output_dir,
            args.registration_method,
            args.threads,
            not args.quiet,
        )
        print(f"LaMAR registration completed in {lamar_time:.2f} seconds")

        # Run direct ANTs registration
        print("\n--- Running direct ANTs registration ---")
        ants_time, ants_output = run_direct_ants_registration(
            args.moving,
            args.fixed,
            output_dir,
            args.registration_method,
            args.threads,
            not args.quiet,
        )
        print(f"Direct ANTs registration completed in {ants_time:.2f} seconds")

        # Compare speeds
        speedup = ants_time / lamar_time if lamar_time > 0 else 0
        if speedup > 1:
            print(f"\nLaMAR is {speedup:.2f}x faster than direct ANTs registration")
        else:
            print(f"\nDirect ANTs is {1/speedup:.2f}x faster than LaMAR")

        # Compare quality
        print("\n--- Comparing registration quality ---")
        try:
            quality_results = compare_registration_quality(
                lamar_output, ants_output, args.fixed
            )

            # Display results for each metric
            for metric_name, scores in quality_results.items():
                if scores:
                    print(f"\n{metric_name.upper()} Metric:")
                    print(f"  LaMAR: {scores['lamar']:.4f}")
                    print(f"  ANTs:  {scores['ants']:.4f}")

                    diff = abs(scores["lamar"] - scores["ants"])
                    print(f"  Difference: {diff:.4f}")

                    if scores["lamar"] > scores["ants"]:
                        print("  LaMAR registration quality is higher")
                    else:
                        print("  Direct ANTs registration quality is higher")
        except Exception as e:
            print(f"Error comparing registration quality: {e}")

        print("\nBenchmark completed successfully!")

    except Exception as e:
        print(f"Error during benchmark: {e}")
        return 1
    finally:
        # Clean up temporary directory if created
        if temp_dir and not args.keep_files:
            shutil.rmtree(temp_dir)
        elif args.keep_files:
            print(f"\nOutput files kept in: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
