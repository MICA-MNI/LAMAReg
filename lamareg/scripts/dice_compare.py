"""
 Dice Score Comparison for Brain Parcellation Maps

This script compares two brain parcellation images and calculates the Dice similarity coefficient for each label (region).

Features:
- Computes Dice score per anatomical label.
- Maps label numbers to region names using FreeSurfer and Desikan-Killiany label conventions.
- Outputs a readable CSV file with label, region name, and Dice score.
- Accepts command-line arguments using argparse:
    --ref : reference parcellation image (e.g., fixed image)
    --reg : registered parcellation image (e.g., moving image after registration)
    --out : output CSV file to save results

This script helps to evaluate the accuracy of image registration or segmentation
by comparing anatomical agreement between two labeled brain volumes.
""" 


import nibabel as nib
import numpy as np
from collections import defaultdict
import csv
import os
import argparse
import sys
from colorama import init, Fore, Style

init()

# FreeSurfer label-to-region mapping
FREESURFER_LABELS = {
    0: "Background", 2: "Left cerebral white matter", 3: "Left cerebral cortex",
    4: "Left lateral ventricle", 5: "Left inferior lateral ventricle",
    7: "Left cerebellum white matter", 8: "Left cerebellum cortex",
    10: "Left thalamus", 11: "Left caudate", 12: "Left putamen", 13: "Left pallidum",
    14: "3rd ventricle", 15: "4th ventricle", 16: "Brain-stem",
    17: "Left hippocampus", 18: "Left amygdala", 24: "CSF", 26: "Left accumbens area",
    28: "Left ventral DC", 41: "Right cerebral white matter", 42: "Right cerebral cortex",
    43: "Right lateral ventricle", 44: "Right inferior lateral ventricle",
    46: "Right cerebellum white matter", 47: "Right cerebellum cortex",
    49: "Right thalamus", 50: "Right caudate", 51: "Right putamen",
    52: "Right pallidum", 53: "Right hippocampus", 54: "Right amygdala",
    58: "Right accumbens area", 60: "Right ventral DC"
}

# Desikan-Killiany cortical labels (1001–1035 = left, 2001–2035 = right)
desikan_labels = [
    "banks STS", "caudal anterior cingulate", "caudal middle frontal", "corpuscallosum", "cuneus",
    "entorhinal", "fusiform", "inferior parietal", "inferior temporal",
    "isthmus cingulate", "lateral occipital", "lateral orbitofrontal",
    "lingual", "medial orbitofrontal", "middle temporal", "parahippocampal",
    "paracentral", "pars opercularis", "pars orbitalis", "pars triangularis",
    "pericalcarine", "postcentral", "posterior cingulate", "precentral",
    "precuneus", "rostral anterior cingulate", "rostral middle frontal",
    "superior frontal", "superior parietal", "superior temporal",
    "supramarginal", "frontal pole", "temporal pole", "transverse temporal", "insula"
]

for i, name in enumerate(desikan_labels):
    FREESURFER_LABELS[1001 + i] = f"Left {name}"
    FREESURFER_LABELS[2001 + i] = f"Right {name}"

def dice_score(mask1, mask2):
    intersection = np.sum((mask1 > 0) & (mask2 > 0))
    size1 = np.sum(mask1 > 0)
    size2 = np.sum(mask2 > 0)
    if size1 + size2 == 0:
        return np.nan
    return 2.0 * intersection / (size1 + size2)

def compare_parcellations_dice(parc1_path, parc2_path, output_csv_path):
    """
    Compare two parcellation images and calculate Dice scores for each label.
    
    Automatically resamples images to the same space if needed using nearest neighbor
    interpolation to preserve label values.
    """
    # Load the parcellation images with nibabel to check dimensions
    parc1_img = nib.load(parc1_path)
    parc2_img = nib.load(parc2_path)
    
    # Check if the images are in the same space
    same_shape = parc1_img.shape == parc2_img.shape
    same_affine = np.allclose(parc1_img.affine, parc2_img.affine, atol=1e-3)
    
    if not (same_shape and same_affine):
        print("\nWARNING: Images are not in the same space.")
        print(f"Image 1 shape: {parc1_img.shape}, Image 2 shape: {parc2_img.shape}")
        print("Resampling to the larger image's space using nearest neighbor interpolation...")
        
        # Determine which image has the larger dimensions (by volume)
        vol1 = np.prod(parc1_img.shape)
        vol2 = np.prod(parc2_img.shape)
        
        # Use ANTs for resampling with nearest neighbor interpolation
        import ants
        
        parc1_ants = ants.image_read(parc1_path)
        parc2_ants = ants.image_read(parc2_path)
        
        if vol1 >= vol2:
            # Resample parc2 to parc1's space
            print(f"Resampling {os.path.basename(parc2_path)} to match {os.path.basename(parc1_path)}")
            resampled = ants.resample_image_to_target(
                parc2_ants, parc1_ants, interp_type='nearestNeighbor')
            parc1 = parc1_ants.numpy()
            parc2 = resampled.numpy()
        else:
            # Resample parc1 to parc2's space
            print(f"Resampling {os.path.basename(parc1_path)} to match {os.path.basename(parc2_path)}")
            resampled = ants.resample_image_to_target(
                parc1_ants, parc2_ants, interp_type='nearestNeighbor')
            parc1 = resampled.numpy()
            parc2 = parc2_ants.numpy()
    else:
        # Images are already in the same space
        parc1 = parc1_img.get_fdata()
        parc2 = parc2_img.get_fdata()
    
    # Convert to integer type to ensure label values are preserved
    parc1 = parc1.astype(int)
    parc2 = parc2.astype(int)
    
    # Continue with the existing code
    labels = sorted(set(np.unique(parc1)) | set(np.unique(parc2)))
    labels = [label for label in labels if label != 0]  # Exclude background

    print(f"\nDice scores per label:\n{'Label':<8}{'Region':<40}{'Dice Score':<10}")
    print("-" * 65)

    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Label", "Region", "Dice Score"])

        for label in labels:
            mask1 = (parc1 == label)
            mask2 = (parc2 == label)
            dice = dice_score(mask1, mask2)
            region = FREESURFER_LABELS.get(label, "Unknown Region")
            print(f"{label:<8}{region:<40}{dice:.4f}")
            writer.writerow([label, region, f"{dice:.4f}"])

    print(f"\nDice scores with region names saved to: {output_csv_path}")

def print_help():
    """Print help message for dice-compare command."""
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
    ║                        DICE COMPARISON                         ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}

    This tool compares two brain parcellation images and calculates the Dice 
    similarity coefficient for each anatomical label. It provides quantitative 
    assessment of registration or segmentation accuracy.

    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      lamar dice-compare {GREEN}[options]{RESET}

    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--ref{RESET} PATH  : Reference parcellation image (.nii.gz)
      {YELLOW}--reg{RESET} PATH  : Registered parcellation image to compare (.nii.gz)
      {YELLOW}--out{RESET} PATH  : Output CSV file for Dice scores (.csv)

    {CYAN}{BOLD}────────────────── EXAMPLE USAGE ────────────────────────{RESET}

    {BLUE}# Calculate Dice scores between reference and registered parcellations{RESET}
    lamar dice-compare {YELLOW}--ref{RESET} fixed_parc.nii.gz {YELLOW}--reg{RESET} registered_parc.nii.gz \\
      {YELLOW}--out{RESET} dice_scores.csv

    {CYAN}{BOLD}────────────────────────── NOTES ───────────────────────{RESET}
    {MAGENTA}•{RESET} Higher Dice scores indicate better spatial overlap between regions
    {MAGENTA}•{RESET} Scores range from 0 (no overlap) to 1 (perfect overlap)
    {MAGENTA}•{RESET} Results include label numbers and anatomical region names
    {MAGENTA}•{RESET} Both FreeSurfer subcortical and Desikan-Killiany cortical regions are supported
    {MAGENTA}•{RESET} Evaluation is performed for each brain region individually
    """
    print(help_text)

def main():
    """Entry point for command-line use"""
    # Check if no arguments were provided or help requested
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help()
        sys.exit(0)
        
    parser = argparse.ArgumentParser(description="Compute Dice score between two parcellation images.")
    parser.add_argument("--ref", required=True, help="Path to reference parcellation image")
    parser.add_argument("--reg", required=True, help="Path to registered parcellation image")
    parser.add_argument("--out", required=True, help="Output CSV file path")
    args = parser.parse_args()

    compare_parcellations_dice(args.ref, args.reg, args.out)

if __name__ == "__main__":
    main()
