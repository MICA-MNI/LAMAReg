#!/usr/bin/env python3
"""
Compute registration quality metrics for existing registration results.

This script traverses directories created by fulltest.py and computes quality metrics
for all registration methods, then outputs the results to a CSV file.
"""

import os
import csv
import argparse
import numpy as np
import nibabel as nib
import torch
import ants
import sys
from torch_mind import MIND3D

def compute_quality_metrics(
    fixed_img,
    lamar_output,
    ants_output,
    lamar_robust_output=None,
    ants_default_output=None
):
    """Compute registration quality metrics for all available outputs."""
    # Load images
    fixed_img_nib = nib.load(fixed_img)
    lamar_img_nib = nib.load(lamar_output)
    ants_img_nib = nib.load(ants_output)
    
    # Convert to numpy arrays
    fixed_img_data = fixed_img_nib.get_fdata()
    lamar_img_data = lamar_img_nib.get_fdata()
    ants_img_data = ants_img_nib.get_fdata()
    
    # Load robust LAMAReg and ANTs default if provided
    lamar_robust_data = None
    lamar_robust_tensor = None
    ants_default_data = None
    ants_default_tensor = None
    
    if lamar_robust_output is not None:
        lamar_robust_nib = nib.load(lamar_robust_output)
        lamar_robust_data = lamar_robust_nib.get_fdata()
        lamar_robust_tensor = torch.from_numpy(lamar_robust_data).float().unsqueeze(0).unsqueeze(0)
        
    if ants_default_output is not None:
        ants_default_nib = nib.load(ants_default_output)
        ants_default_data = ants_default_nib.get_fdata()
        ants_default_tensor = torch.from_numpy(ants_default_data).float().unsqueeze(0).unsqueeze(0)
    
    # Convert to PyTorch tensors
    lamar_tensor = torch.from_numpy(lamar_img_data).float().unsqueeze(0).unsqueeze(0)
    ants_tensor = torch.from_numpy(ants_img_data).float().unsqueeze(0).unsqueeze(0)
    fixed_tensor = torch.from_numpy(fixed_img_data).float().unsqueeze(0).unsqueeze(0)
    
    results = {}
    
    # Calculate Mutual Information using ANTsPy
    def mutual_information(img1, img2):
        """Calculate mutual information between two images using ANTsPy."""
        # Convert numpy arrays to ANTs images
        img1_ants = ants.from_numpy(img1.astype(np.float32))
        img2_ants = ants.from_numpy(img2.astype(np.float32))
        
        # Calculate mutual information directly with ANTsPy
        return ants.image_mutual_information(img1_ants, img2_ants)

    # Calculate ANTSNeighborhoodCorrelation using ANTsPy
    def ants_neighborhood_correlation(img1, img2):
        """Calculate ANTSNeighborhoodCorrelation between two images using ANTsPy."""
        # Convert numpy arrays to ANTs images
        img1_ants = ants.from_numpy(img1.astype(np.float32))
        img2_ants = ants.from_numpy(img2.astype(np.float32))
        
        # Use ANTSNeighborhoodCorrelation metric directly
        similarity = ants.image_similarity(img1_ants, img2_ants, metric_type='ANTSNeighborhoodCorrelation')
        
        return similarity
    
    # Calculate and store MI
    results["mi"] = {
        "lamar": mutual_information(lamar_img_data, fixed_img_data),
        "ants": mutual_information(ants_img_data, fixed_img_data),
    }
    
    # Calculate and store ANTSNeighborhoodCorrelation
    results["antsneighborhoodcorrelation"] = {
        "lamar": ants_neighborhood_correlation(lamar_img_data, fixed_img_data),
        "ants": ants_neighborhood_correlation(ants_img_data, fixed_img_data),
    }
    
    # Add robust and default results if available
    if lamar_robust_data is not None:
        results["mi"]["lamar_robust"] = mutual_information(lamar_robust_data, fixed_img_data)
        results["antsneighborhoodcorrelation"]["lamar_robust"] = ants_neighborhood_correlation(lamar_robust_data, fixed_img_data)
        
    if ants_default_data is not None:
        results["mi"]["ants_default"] = mutual_information(ants_default_data, fixed_img_data)
        results["antsneighborhoodcorrelation"]["ants_default"] = ants_neighborhood_correlation(ants_default_data, fixed_img_data)
    
    # Calculate MIND metric - Using exact implementation from runtest.py
    try:
        from torch_mind import MINDLoss3D

        mind_loss = MINDLoss3D()

        with torch.no_grad():
            lamar_mind = mind_loss(lamar_tensor, fixed_tensor).item()
            ants_mind = mind_loss(ants_tensor, fixed_tensor).item()
            
            mind_results = {"lamar": lamar_mind, "ants": ants_mind}
            
            if lamar_robust_tensor is not None:
                mind_results["lamar_robust"] = mind_loss(lamar_robust_tensor, fixed_tensor).item()
            
            if ants_default_tensor is not None:
                mind_results["ants_default"] = mind_loss(ants_default_tensor, fixed_tensor).item()
            
            results["mind"] = mind_results
    except Exception as e:
        print(f"Error calculating MIND: {e}")
        results["mind"] = None

    # Calculate NGF metric - Using exact implementation from runtest.py
    try:
        from normalized_gradient_field import NormalizedGradientField3d

        pixel_spacing = lamar_img_nib.header.get_zooms()[:3]

        ngf = NormalizedGradientField3d(
            grad_method="default",
            mm_spacing=pixel_spacing,
            reduction="mean",
        )

        with torch.no_grad():
            lamar_ngf = ngf(lamar_tensor, fixed_tensor).item()
            ants_ngf = ngf(ants_tensor, fixed_tensor).item()
            
            ngf_results = {"lamar": lamar_ngf, "ants": ants_ngf}
            
            if lamar_robust_tensor is not None:
                ngf_results["lamar_robust"] = ngf(lamar_robust_tensor, fixed_tensor).item()
                
            if ants_default_tensor is not None:
                ngf_results["ants_default"] = ngf(ants_default_tensor, fixed_tensor).item()
            
            results["ngf"] = ngf_results
    except Exception as e:
        print(f"Error calculating NGF: {e}")
        results["ngf"] = None
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Compute registration quality metrics for existing registration results"
    )
    parser.add_argument(
        "--data-path",
        default="/host/verges/tank/data/ian/MICs_MF_Diffusion",
        help="Path to BIDS dataset with original images"
    )
    parser.add_argument(
        "--results-dir",
        default="./registration_results",
        help="Directory containing registration results from fulltest.py"
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV file path (default: <results_dir>/metrics_results.csv)"
    )
    args = parser.parse_args()
    
    # Set default output CSV if not specified
    if args.output_csv is None:
        args.output_csv = os.path.join(args.results_dir, "metrics_results.csv")
    
    # Define CSV fieldnames
    fieldnames = [
        "subject_id",
        "session_id",
        "mi_lamar",
        "mi_lamar_robust",
        "mi_ants",
        "mi_ants_default",
        "antsneighborhoodcorrelation_lamar",
        "antsneighborhoodcorrelation_lamar_robust",
        "antsneighborhoodcorrelation_ants",
        "antsneighborhoodcorrelation_ants_default",
        "mind_lamar",
        "mind_lamar_robust",
        "mind_ants",
        "mind_ants_default",
        "ngf_lamar",
        "ngf_lamar_robust",
        "ngf_ants",
        "ngf_ants_default",
    ]
    
    # Create CSV file with headers
    with open(args.output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # Process directories
    for item in os.listdir(args.results_dir):
        item_path = os.path.join(args.results_dir, item)
        
        # Skip non-directories and the main results CSV file
        if not os.path.isdir(item_path) or item == "mind_descriptors":
            continue
        
        try:
            # Extract subject and session info from directory name
            if "_" in item:
                subject_id, session_id = item.split("_", 1)
            else:
                # Skip if not in the expected format
                continue
            
            # Standard filenames from fulltest.py
            lamar_output = os.path.join(item_path, "lamar_registered.nii.gz")
            lamar_robust_output = os.path.join(item_path, "lamar_robust_registered.nii.gz")
            ants_output = os.path.join(item_path, "direct_ants_registered.nii.gz")
            ants_default_output = os.path.join(item_path, "direct_ants_default_registered.nii.gz")
        
            # Skip if any of the required files are missing
            if not os.path.exists(lamar_output):
                print(f"  Skipping {item}: missing required registration outputs")
                print(f"    Expected: {lamar_output}")
                continue
            if not os.path.exists(ants_output):
                print(f"  Skipping {item}: missing required registration outputs")
                print(f"    Expected: {ants_output}")
                continue
            # Find the T1w reference image
            t1w_file = os.path.join(
                args.data_path, 
                subject_id, 
                session_id, 
                "anat",
                f"{subject_id}_{session_id}_T1w-space_T1w.nii.gz"
            )
            
            if not os.path.exists(t1w_file):
                print(f"  Skipping {item}: T1w reference file not found at {t1w_file}")
                continue
            
            print(f"Processing {item}...")
            
            # Check if optional files exist
            lamar_robust = lamar_robust_output if os.path.exists(lamar_robust_output) else None
            ants_default = ants_default_output if os.path.exists(ants_default_output) else None
            
            # Compute quality metrics
            metrics = compute_quality_metrics(
                fixed_img=t1w_file,
                lamar_output=lamar_output,
                ants_output=ants_output,
                lamar_robust_output=lamar_robust,
                ants_default_output=ants_default
            )
            
            # Extract metrics
            mi = metrics.get("mi", {})
            antsneighborhoodcorrelation = metrics.get("antsneighborhoodcorrelation", {})
            mind = metrics.get("mind", {})
            ngf = metrics.get("ngf", {})
            
            # Create row data
            row_data = {
                "subject_id": subject_id,
                "session_id": session_id,
                "mi_lamar": f"{mi.get('lamar', 'N/A')}" if mi else "N/A",
                "mi_lamar_robust": f"{mi.get('lamar_robust', 'N/A')}" if mi and 'lamar_robust' in mi else "N/A",
                "mi_ants": f"{mi.get('ants', 'N/A')}" if mi else "N/A",
                "mi_ants_default": f"{mi.get('ants_default', 'N/A')}" if mi and 'ants_default' in mi else "N/A",
                "antsneighborhoodcorrelation_lamar": f"{antsneighborhoodcorrelation.get('lamar', 'N/A')}" if antsneighborhoodcorrelation else "N/A",
                "antsneighborhoodcorrelation_lamar_robust": f"{antsneighborhoodcorrelation.get('lamar_robust', 'N/A')}" if antsneighborhoodcorrelation and 'lamar_robust' in antsneighborhoodcorrelation else "N/A",
                "antsneighborhoodcorrelation_ants": f"{antsneighborhoodcorrelation.get('ants', 'N/A')}" if antsneighborhoodcorrelation else "N/A",
                "antsneighborhoodcorrelation_ants_default": f"{antsneighborhoodcorrelation.get('ants_default', 'N/A')}" if antsneighborhoodcorrelation and 'ants_default' in antsneighborhoodcorrelation else "N/A",
                "mind_lamar": f"{mind.get('lamar', 'N/A')}" if mind else "N/A",
                "mind_lamar_robust": f"{mind.get('lamar_robust', 'N/A')}" if mind and 'lamar_robust' in mind else "N/A",
                "mind_ants": f"{mind.get('ants', 'N/A')}" if mind else "N/A",
                "mind_ants_default": f"{mind.get('ants_default', 'N/A')}" if mind and 'ants_default' in mind else "N/A",
                "ngf_lamar": f"{ngf.get('lamar', 'N/A')}" if ngf else "N/A",
                "ngf_lamar_robust": f"{ngf.get('lamar_robust', 'N/A')}" if ngf and 'lamar_robust' in ngf else "N/A",
                "ngf_ants": f"{ngf.get('ants', 'N/A')}" if ngf else "N/A",
                "ngf_ants_default": f"{ngf.get('ants_default', 'N/A')}" if ngf and 'ants_default' in ngf else "N/A",
            }
            
            # Write results to CSV in append mode
            with open(args.output_csv, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(row_data)
            
            # Print summary
            print(f"  Quality metrics computed for {item}")
            print(f"    MI: LaMAR={mi.get('lamar', 'N/A'):.4f}, LaMAR Robust={mi.get('lamar_robust', 'N/A') if 'lamar_robust' in mi else 'N/A'}")
            print(f"        ANTs={mi.get('ants', 'N/A'):.4f}, ANTs Default={mi.get('ants_default', 'N/A') if 'ants_default' in mi else 'N/A'}")
            print(f"    ANTS Neighborhood Correlation: LaMAR={antsneighborhoodcorrelation.get('lamar', 'N/A'):.4f}, "
                  f"LaMAR Robust={antsneighborhoodcorrelation.get('lamar_robust', 'N/A') if 'lamar_robust' in antsneighborhoodcorrelation else 'N/A'}")
            print(f"        ANTs={antsneighborhoodcorrelation.get('ants', 'N/A'):.4f}, "
                  f"ANTs Default={antsneighborhoodcorrelation.get('ants_default', 'N/A') if 'ants_default' in antsneighborhoodcorrelation else 'N/A'}")
            print(f"    MIND: LaMAR={mind.get('lamar', 'N/A'):.4f}, "
                  f"LaMAR Robust={mind.get('lamar_robust', 'N/A') if 'lamar_robust' in mind else 'N/A'}")
            print(f"        ANTs={mind.get('ants', 'N/A'):.4f}, "
                  f"ANTs Default={mind.get('ants_default', 'N/A') if 'ants_default' in mind else 'N/A'}")
            print(f"    NGF: LaMAR={ngf.get('lamar', 'N/A'):.4f}, "
                  f"LaMAR Robust={ngf.get('lamar_robust', 'N/A') if 'lamar_robust' in ngf else 'N/A'}")
            print(f"        ANTs={ngf.get('ants', 'N/A'):.4f}, "
                  f"ANTs Default={ngf.get('ants_default', 'N/A') if 'ants_default' in ngf else 'N/A'}")
        except Exception as e:
            print(f"  Error processing {item}: {e}")
            continue
    
    print(f"\nMetrics calculation complete. Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()