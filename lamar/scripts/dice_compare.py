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
    parc1 = nib.load(parc1_path).get_fdata().astype(int)
    parc2 = nib.load(parc2_path).get_fdata().astype(int)

    labels = sorted(set(np.unique(parc1)) | set(np.unique(parc2)))
    labels = [label for label in labels if label != 0]

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

    print(f"\n Dice scores with region names saved to: {output_csv_path}")

def print_help():
    """Print help message for dice-compare command."""
    help_text = """
    Dice Compare: Calculate Dice Similarity Metrics for Brain Parcellations
    ---------------------------------------------------------------------
    
    This tool compares two brain parcellation images and calculates the Dice 
    similarity coefficient for each anatomical label.
    
    Usage:
      dice-compare --ref REFERENCE_PARCELLATION --reg REGISTERED_PARCELLATION --out OUTPUT_CSV
    
    Required Arguments:
      --ref PATH          Reference parcellation image
      --reg PATH          Registered parcellation image to compare
      --out PATH          Output CSV file for results
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
