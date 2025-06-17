#!/usr/bin/env python
import os
import sys
import numpy as np
import nibabel as nib
import argparse
import torch
from typing import Optional, Tuple, Dict, Any


def load_nifti(file_path: str) -> Tuple[np.ndarray, Any]:
    """Load a NIfTI file and return its data."""
    try:
        img = nib.load(file_path)
        return img.get_fdata(), img.header
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise


def calculate_mutual_information(
    img1_data: np.ndarray,
    img2_data: np.ndarray,
    mask_data: Optional[np.ndarray] = None,
    bins: int = 32,
) -> float:
    """
    Calculate mutual information between two images using histogram method.

    Parameters:
    - img1_data, img2_data: 3D numpy arrays containing image data
    - mask_data: Optional binary mask to restrict calculation to specific voxels
    - bins: number of bins for histogram calculation

    Returns:
    - Mutual information value
    """
    # Flatten the images to 1D arrays
    img1_flat = img1_data.flatten()
    img2_flat = img2_data.flatten()

    # Apply mask if provided
    if mask_data is not None:
        mask_flat = mask_data.flatten().astype(bool)
        img1_flat = img1_flat[mask_flat]
        img2_flat = img2_flat[mask_flat]

    # Calculate joint histogram
    joint_hist, _, _ = np.histogram2d(img1_flat, img2_flat, bins=bins)

    # Normalize the histogram to get joint probability
    joint_prob = joint_hist / np.sum(joint_hist)

    # Calculate marginal probabilities
    img1_prob = np.sum(joint_prob, axis=1)
    img2_prob = np.sum(joint_prob, axis=0)

    # Calculate entropy of each image and joint entropy
    img1_entropy = -np.sum(img1_prob * np.log2(img1_prob + np.finfo(float).eps))
    img2_entropy = -np.sum(img2_prob * np.log2(img2_prob + np.finfo(float).eps))

    # Calculate joint entropy
    joint_entropy = -np.sum(joint_prob * np.log2(joint_prob + np.finfo(float).eps))

    # Calculate mutual information
    mutual_info = img1_entropy + img2_entropy - joint_entropy

    return mutual_info


def calculate_normalized_mutual_information(
    img1_data: np.ndarray,
    img2_data: np.ndarray,
    mask_data: Optional[np.ndarray] = None,
    bins: int = 32,
) -> float:
    """
    Calculate normalized mutual information between two images using histogram method.

    Parameters:
    - img1_data, img2_data: 3D numpy arrays containing image data
    - mask_data: Optional binary mask to restrict calculation to specific voxels
    - bins: number of bins for histogram calculation

    Returns:
    - Normalized mutual information value (ranges from 1.0 to 2.0)
    """
    # Flatten the images to 1D arrays
    img1_flat = img1_data.flatten()
    img2_flat = img2_data.flatten()

    # Apply mask if provided
    if mask_data is not None:
        mask_flat = mask_data.flatten().astype(bool)
        img1_flat = img1_flat[mask_flat]
        img2_flat = img2_flat[mask_flat]

    # Calculate joint histogram
    joint_hist, _, _ = np.histogram2d(img1_flat, img2_flat, bins=bins)

    # Normalize the histogram to get joint probability
    joint_prob = joint_hist / np.sum(joint_hist)

    # Calculate marginal probabilities
    img1_prob = np.sum(joint_prob, axis=1)
    img2_prob = np.sum(joint_prob, axis=0)

    # Calculate entropy of each image and joint entropy
    img1_entropy = -np.sum(img1_prob * np.log2(img1_prob + np.finfo(float).eps))
    img2_entropy = -np.sum(img2_prob * np.log2(img2_prob + np.finfo(float).eps))

    # Calculate joint entropy
    joint_entropy = -np.sum(joint_prob * np.log2(joint_prob + np.finfo(float).eps))

    # Calculate normalized mutual information (Studholme et al. 1999)
    nmi = (img1_entropy + img2_entropy) / joint_entropy if joint_entropy > 0 else 0

    return nmi


def calculate_normalized_gradient_field(
    img1_data: np.ndarray,
    img2_data: np.ndarray,
    mask_data: Optional[np.ndarray] = None,
    header: Optional[Any] = None,
) -> Optional[float]:
    """
    Calculate normalized gradient field similarity between two images.

    Parameters:
    - img1_data, img2_data: 3D numpy arrays containing image data
    - mask_data: Optional binary mask to restrict calculation to specific voxels
    - header: NIfTI header to get voxel spacing

    Returns:
    - NGF similarity value (raw value, not negated)
    """
    try:
        from normalized_gradient_field import NormalizedGradientField3d

        # Convert to PyTorch tensors
        img1_tensor = torch.from_numpy(img1_data).float().unsqueeze(0).unsqueeze(0)
        img2_tensor = torch.from_numpy(img2_data).float().unsqueeze(0).unsqueeze(0)

        # Get pixel spacing from header if available
        mm_spacing = None
        if header is not None:
            mm_spacing = header.get_zooms()[:3]

        # Create NGF calculator
        ngf_calc = NormalizedGradientField3d(
            grad_method="default",
            gauss_sigma=0.5,
            eps=1e-5,
            mm_spacing=mm_spacing,
            reduction="mean",
        )

        # Calculate NGF (return raw value)
        with torch.no_grad():
            ngf_value = ngf_calc(img1_tensor, img2_tensor).item()

        return ngf_value

    except ImportError:
        print(
            "Warning: Could not import NormalizedGradientField3d. Skipping NGF calculation."
        )
        return None
    except Exception as e:
        print(f"Error calculating NGF: {e}")
        return None


def calculate_mind(
    img1_data: np.ndarray, img2_data: np.ndarray, mask_data: Optional[np.ndarray] = None
) -> Optional[float]:
    """
    Calculate MIND similarity between two images.

    Parameters:
    - img1_data, img2_data: 3D numpy arrays containing image data
    - mask_data: Optional binary mask (not used directly in MIND calculation)

    Returns:
    - MIND similarity value (raw value, not negated)
    """
    try:
        from torch_mind import MINDLoss3D

        # Convert to PyTorch tensors
        img1_tensor = torch.from_numpy(img1_data).float().unsqueeze(0).unsqueeze(0)
        img2_tensor = torch.from_numpy(img2_data).float().unsqueeze(0).unsqueeze(0)

        # Create MIND calculator
        mind_loss = MINDLoss3D(
            patch_size=3,
            sigma=0.5,
        )

        # Calculate MIND (return raw value)
        with torch.no_grad():
            mind_value = mind_loss(img1_tensor, img2_tensor).item()

        return mind_value

    except ImportError:
        print("Warning: Could not import MINDLoss3D. Skipping MIND calculation.")
        return None
    except Exception as e:
        print(f"Error calculating MIND: {e}")
        return None


def calculate_all_metrics(
    file1: str,
    file2: str,
    mask_file: Optional[str] = None,
    bins: int = 32,
    skip_ngf: bool = False,
    skip_mind: bool = False,
) -> Dict[str, Optional[float]]:
    """
    Calculate all similarity metrics between two NIfTI files.

    Parameters:
    - file1, file2: paths to NIfTI files
    - mask_file: optional path to mask NIfTI file
    - bins: number of bins for histogram calculation
    - skip_ngf: whether to skip NGF calculation
    - skip_mind: whether to skip MIND calculation

    Returns:
    - Dictionary with metric names as keys and values as results
    """
    # Validate files exist
    if not os.path.exists(file1):
        raise FileNotFoundError(f"File not found: {file1}")
    if not os.path.exists(file2):
        raise FileNotFoundError(f"File not found: {file2}")

    # Load images
    img1_data, img1_header = load_nifti(file1)
    img2_data, img2_header = load_nifti(file2)

    # Load mask if provided
    mask_data = None
    if mask_file:
        if not os.path.exists(mask_file):
            raise FileNotFoundError(f"Mask file not found: {mask_file}")
        mask_data, _ = load_nifti(mask_file)

        # Check mask dimensions
        if mask_data.shape != img1_data.shape:
            raise ValueError(
                f"Mask shape {mask_data.shape} doesn't match image shape {img1_data.shape}"
            )

    # Check image dimensions match
    if img1_data.shape != img2_data.shape:
        raise ValueError(
            f"Image shapes don't match: {img1_data.shape} vs {img2_data.shape}"
        )

    # Calculate metrics
    results = {}

    # Always calculate MI and NMI
    results["mi"] = calculate_mutual_information(img1_data, img2_data, mask_data, bins)
    results["nmi"] = calculate_normalized_mutual_information(
        img1_data, img2_data, mask_data, bins
    )

    # Calculate NGF if requested
    if not skip_ngf:
        results["ngf"] = calculate_normalized_gradient_field(
            img1_data, img2_data, mask_data, img1_header
        )
    else:
        results["ngf"] = None

    # Calculate MIND if requested
    if not skip_mind:
        results["mind"] = calculate_mind(img1_data, img2_data, mask_data)
    else:
        results["mind"] = None

    return results


def calculate_metrics_from_arrays(
    img1_data: np.ndarray,
    img2_data: np.ndarray,
    mask_data: Optional[np.ndarray] = None,
    header: Optional[Any] = None,
    bins: int = 32,
    skip_ngf: bool = False,
    skip_mind: bool = False,
) -> Dict[str, Optional[float]]:
    """
    Calculate all similarity metrics from numpy arrays directly.

    Parameters:
    - img1_data, img2_data: 3D numpy arrays containing image data
    - mask_data: optional binary mask array
    - header: optional NIfTI header for voxel spacing
    - bins: number of bins for histogram calculation
    - skip_ngf: whether to skip NGF calculation
    - skip_mind: whether to skip MIND calculation

    Returns:
    - Dictionary with metric names as keys and values as results
    """
    # Check image dimensions match
    if img1_data.shape != img2_data.shape:
        raise ValueError(
            f"Image shapes don't match: {img1_data.shape} vs {img2_data.shape}"
        )

    # Check mask dimensions if provided
    if mask_data is not None and mask_data.shape != img1_data.shape:
        raise ValueError(
            f"Mask shape {mask_data.shape} doesn't match image shape {img1_data.shape}"
        )

    # Calculate metrics
    results = {}

    # Always calculate MI and NMI
    results["mi"] = calculate_mutual_information(img1_data, img2_data, mask_data, bins)
    results["nmi"] = calculate_normalized_mutual_information(
        img1_data, img2_data, mask_data, bins
    )

    # Calculate NGF if requested
    if not skip_ngf:
        results["ngf"] = calculate_normalized_gradient_field(
            img1_data, img2_data, mask_data, header
        )
    else:
        results["ngf"] = None

    # Calculate MIND if requested
    if not skip_mind:
        results["mind"] = calculate_mind(img1_data, img2_data, mask_data)
    else:
        results["mind"] = None

    return results


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Calculate image similarity metrics between two NIfTI files."
    )
    parser.add_argument("file1", help="First NIfTI file")
    parser.add_argument("file2", help="Second NIfTI file")
    parser.add_argument("--mask", help="Optional mask NIfTI file")
    parser.add_argument(
        "--bins",
        type=int,
        default=32,
        help="Number of bins for histogram calculation (default: 32)",
    )
    parser.add_argument(
        "--skip-ngf",
        action="store_true",
        help="Skip normalized gradient field calculation",
    )
    parser.add_argument(
        "--skip-mind",
        action="store_true",
        help="Skip MIND calculation",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the results, no progress messages",
    )
    args = parser.parse_args()

    try:
        # Calculate all metrics
        results = calculate_all_metrics(
            args.file1,
            args.file2,
            args.mask,
            bins=args.bins,
            skip_ngf=args.skip_ngf,
            skip_mind=args.skip_mind,
        )

        # Print results
        if not args.quiet:
            print("\nSimilarity Metrics:")
            print("=" * 40)

        print(f"Mutual Information: {results['mi']:.6f}")
        print(f"Normalized Mutual Information: {results['nmi']:.6f}")

        if results["ngf"] is not None:
            print(f"Normalized Gradient Field: {results['ngf']:.6f}")

        if results["mind"] is not None:
            print(f"MIND similarity: {results['mind']:.6f}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
