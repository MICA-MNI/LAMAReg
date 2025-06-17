# SyNRA results:

# SUMMARY COMPARISON
# ============================================================
# Registration Method          |    MI    |   NMI    |   NGF    |  MIND
# ----------------------------------------------------------------------
# Full (100,50,30)            |   0.5040 |   1.1238 |  -0.1209 |   0.0903
# Half (50,30)                |   0.4817 |   1.1197 |  -0.1198 |   0.0900
# Single (30)                 |   0.4367 |   1.1073 |  -0.1174 |   0.0901
# Missing last (50,0)         |   0.4367 |   1.1073 |  -0.1174 |   0.0901

# SyNONLY results:
# ============================================================
# SUMMARY COMPARISON
# ============================================================
# Registration Method          |    MI    |   NMI    |   NGF    |  MIND
# ----------------------------------------------------------------------
# Full (100,50,30)            |   0.4888 |   1.1198 |  -0.1193 |   0.0904
# Half (50,30)                |   0.4707 |   1.1177 |  -0.1209 |   0.0897
# Single (30)                 |   0.4261 |   1.1049 |  -0.1209 |   0.0897
# Missing last (50,0)         |   0.4261 |   1.1049 |  -0.1209 |   0.0897

# SyN
# ============================================================
# SUMMARY COMPARISON
# ============================================================
# Registration Method          |    MI    |   NMI    |   NGF    |  MIND
# ----------------------------------------------------------------------
# Full (100,50,30)            |   0.5055 |   1.1236 |  -0.1209 |   0.0904
# Half (50,30)                |   0.4828 |   1.1199 |  -0.1203 |   0.0899
# Single (30)                 |   0.4387 |   1.1078 |  -0.1182 |   0.0901
# Missing last (50,0)         |   0.4387 |   1.1078 |  -0.1182 |   0.0901

import ants
from calcmi import calculate_all_metrics
import os

# Load images
fixed = ants.image_read("output/sub-HC001_ses-01_T1w.nii.gz")
moving = ants.image_read("output\sub-HC001_ses-02_space-dwi_desc-b0.nii.gz")
initial_transforms = ["output\dwi_to_T1w_warp.nii.gz", "output\dwi_to_T1w_affine.mat"]
# Set ANTs/ITK thread count

env = os.environ.copy()
env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(16)
env["OMP_NUM_THREADS"] = str(16)  # OpenMP threads for ANTs


# # First registration: Full iterations
# print("\n" + "=" * 60)
# print("REGISTRATION 1: Full iterations (100, 50, 30)")
# print("=" * 60)

# transforms = ants.registration(
#     fixed=fixed,
#     moving=moving,
#     type_of_transform="SyNOnly",
#     reg_iterations=(100, 50, 30),
#     initial_transform=initial_transforms,
# )

# ants.image_write(transforms["warpedmovout"], "registered_full.nii.gz")

# # Calculate metrics for first registration
# print("\nMetrics for Full Registration:")
# try:
#     results_full = calculate_all_metrics(
#         "output/sub-HC001_ses-01_T1w.nii.gz",
#         "registered_full.nii.gz",
#         skip_ngf=False,
#         skip_mind=False,
#     )

#     print(f"  MI:   {results_full['mi']:.6f}")
#     print(f"  NMI:  {results_full['nmi']:.6f}")
#     if results_full["ngf"] is not None:
#         print(f"  NGF:  {results_full['ngf']:.6f}")
#     if results_full["mind"] is not None:
#         print(f"  MIND: {results_full['mind']:.6f}")

# except Exception as e:
#     print(f"Error calculating metrics: {e}")
#     results_full = {"mi": 0, "nmi": 0, "ngf": 0, "mind": 0}

# # Second registration: Half iterations
# print("\n" + "=" * 60)
# print("REGISTRATION 2: Half iterations (50, 30)")
# print("=" * 60)

# transforms = ants.registration(
#     fixed=fixed,
#     moving=moving,
#     type_of_transform="SyNOnly",
#     reg_iterations=(50, 30),
#     initial_transform=initial_transforms,
# )

# ants.image_write(transforms["warpedmovout"], "registered_half.nii.gz")

# # Calculate metrics for second registration
# print("\nMetrics for Half Registration:")
# try:
#     results_half = calculate_all_metrics(
#         "output/sub-HC001_ses-01_T1w.nii.gz",
#         "registered_half.nii.gz",
#         skip_ngf=False,
#         skip_mind=False,
#     )

#     print(f"  MI:   {results_half['mi']:.6f}")
#     print(f"  NMI:  {results_half['nmi']:.6f}")
#     if results_half["ngf"] is not None:
#         print(f"  NGF:  {results_half['ngf']:.6f}")
#     if results_half["mind"] is not None:
#         print(f"  MIND: {results_half['mind']:.6f}")

# except Exception as e:
#     print(f"Error calculating metrics: {e}")
#     results_half = {"mi": 0, "nmi": 0, "ngf": 0, "mind": 0}

# Third registration: Single level
print("\n" + "=" * 60)
print("REGISTRATION 3: Single level (50)")
print("=" * 60)

transforms = ants.registration(
    fixed=fixed,
    moving=moving,
    type_of_transform="SyNOnly",
    reg_iterations=(10, 20),
    initial_transform=initial_transforms,
)

ants.image_write(transforms["warpedmovout"], "registered_last.nii.gz")

# Calculate metrics for third registration
print("\nMetrics for Single Level Registration:")
try:
    results_last = calculate_all_metrics(
        "output/sub-HC001_ses-01_T1w.nii.gz",
        "registered_last.nii.gz",
        skip_ngf=False,
        skip_mind=False,
    )

    print(f"  MI:   {results_last['mi']:.6f}")
    print(f"  NMI:  {results_last['nmi']:.6f}")
    if results_last["ngf"] is not None:
        print(f"  NGF:  {results_last['ngf']:.6f}")
    if results_last["mind"] is not None:
        print(f"  MIND: {results_last['mind']:.6f}")

except Exception as e:
    print(f"Error calculating metrics: {e}")
    results_last = {"mi": 0, "nmi": 0, "ngf": 0, "mind": 0}

# Third registration: Single level
print("\n" + "=" * 60)
print("REGISTRATION 3: Single level (30)")
print("=" * 60)

transforms = ants.registration(
    fixed=fixed,
    moving=moving,
    type_of_transform="SyNOnly",
    reg_iterations=(
        50,
        0,
    ),
    initial_transform=initial_transforms,
)

ants.image_write(transforms["warpedmovout"], "registered_missinglast.nii.gz")

# Calculate metrics for third registration
print("\nMetrics for no last Single Level Registration:")
try:
    results_missinglast = calculate_all_metrics(
        "output/sub-HC001_ses-01_T1w.nii.gz",
        "registered_missinglast.nii.gz",
        skip_ngf=False,
        skip_mind=False,
    )

    print(f"  MI:   {results_missinglast['mi']:.6f}")
    print(f"  NMI:  {results_missinglast['nmi']:.6f}")
    if results_missinglast["ngf"] is not None:
        print(f"  NGF:  {results_missinglast['ngf']:.6f}")
    if results_missinglast["mind"] is not None:
        print(f"  MIND: {results_missinglast['mind']:.6f}")

except Exception as e:
    print(f"Error calculating metrics: {e}")
    results_missinglast = {"mi": 0, "nmi": 0, "ngf": 0, "mind": 0}


# Summary comparison
print("\n" + "=" * 60)
print("SUMMARY COMPARISON")
print("=" * 60)

try:
    print("Registration Method          |    MI    |   NMI    |   NGF    |  MIND")
    print("-" * 70)

    # Full registration
    mi_full = results_full["mi"] if results_full["mi"] is not None else 0
    nmi_full = results_full["nmi"] if results_full["nmi"] is not None else 0
    ngf_full = results_full["ngf"] if results_full["ngf"] is not None else 0
    mind_full = results_full["mind"] if results_full["mind"] is not None else 0

    print(
        f"Full (100,50,30)            | {mi_full:8.4f} | {nmi_full:8.4f} | {ngf_full:8.4f} | {mind_full:8.4f}"
    )

    # Half registration
    mi_half = results_half["mi"] if results_half["mi"] is not None else 0
    nmi_half = results_half["nmi"] if results_half["nmi"] is not None else 0
    ngf_half = results_half["ngf"] if results_half["ngf"] is not None else 0
    mind_half = results_half["mind"] if results_half["mind"] is not None else 0

    print(
        f"Half (50,30)                | {mi_half:8.4f} | {nmi_half:8.4f} | {ngf_half:8.4f} | {mind_half:8.4f}"
    )

    # Single level registration
    mi_last = results_last["mi"] if results_last["mi"] is not None else 0
    nmi_last = results_last["nmi"] if results_last["nmi"] is not None else 0
    ngf_last = results_last["ngf"] if results_last["ngf"] is not None else 0
    mind_last = results_last["mind"] if results_last["mind"] is not None else 0

    print(
        f"Single (30)                 | {mi_last:8.4f} | {nmi_last:8.4f} | {ngf_last:8.4f} | {mind_last:8.4f}"
    )

    # No last registration
    mi_missinglast = (
        results_missinglast["mi"] if results_missinglast["mi"] is not None else 0
    )
    nmi_missinglast = (
        results_missinglast["nmi"] if results_missinglast["nmi"] is not None else 0
    )
    ngf_missinglast = (
        results_missinglast["ngf"] if results_missinglast["ngf"] is not None else 0
    )
    mind_missinglast = (
        results_missinglast["mind"] if results_missinglast["mind"] is not None else 0
    )
    print(
        f"Missing last (50,0)         | {mi_missinglast:8.4f} | {nmi_missinglast:8.4f} | {ngf_missinglast:8.4f} | {mind_missinglast:8.4f}"
    )

    # Find best performing registration for each metric
    print("\nBest performing registration:")

    # MI (higher is better)
    mi_values = [mi_full, mi_half, mi_last]
    mi_names = ["Initial", "Full", "Half", "Single"]
    best_mi_idx = mi_values.index(max(mi_values))
    print(f"  Best MI:   {mi_names[best_mi_idx]} ({mi_values[best_mi_idx]:.6f})")

    # NMI (higher is better)
    nmi_values = [nmi_full, nmi_half, nmi_last]
    best_nmi_idx = nmi_values.index(max(nmi_values))
    print(f"  Best NMI:  {mi_names[best_nmi_idx]} ({nmi_values[best_nmi_idx]:.6f})")

    # NGF (more negative is better, so we look for minimum)
    if not all(x == 0 for x in [ngf_full, ngf_half, ngf_last]):
        ngf_values = [ngf_full, ngf_half, ngf_last]
        best_ngf_idx = ngf_values.index(min(ngf_values))
        print(f"  Best NGF:  {mi_names[best_ngf_idx]} ({ngf_values[best_ngf_idx]:.6f})")

    # MIND (lower is better in loss terms)
    if not all(x == 0 for x in [mind_full, mind_half, mind_last]):
        mind_values = [mind_full, mind_half, mind_last]
        best_mind_idx = mind_values.index(min(mind_values))
        print(
            f"  Best MIND: {mi_names[best_mind_idx]} ({mind_values[best_mind_idx]:.6f})"
        )


except Exception as e:
    print(f"Error in summary comparison: {e}")

print("\nRegistration comparison complete!")
