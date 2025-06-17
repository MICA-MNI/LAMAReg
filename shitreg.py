# Registration Method          |    MI    |   NMI    |   NGF    |  MIND
# ----------------------------------------------------------------------
# Default (50,30)             |   0.4585 |   1.1132 |  -0.1158 |   0.0903
# Normaffine (50,30)          |   0.4675 |   1.1169 |  -0.1172 |   0.0897
# Full (100,50,30)            |   0.5035 |   1.1246 |  -0.1188 |   0.0901
# Half (50,30)                |   0.4698 |   1.1172 |  -0.1171 |   0.0898
# Single (30)                 |   0.4328 |   1.1078 |  -0.1150 |   0.0901


import ants
from calcmi import calculate_all_metrics
import os

# Load images
fixed = ants.image_read("output/sub-HC001_ses-01_T1w.nii.gz")
moving = ants.image_read("output/sub-HC001_ses-02_space-dwi_desc-b0.nii.gz")
# Set ANTs/ITK thread count

env = os.environ.copy()
env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(12)
env["OMP_NUM_THREADS"] = str(12)  # OpenMP threads for ANTs

# Second registration: Half iterations
print("\n" + "=" * 60)
print("REGISTRATION 1: Default iterations (50, 30)")
print("=" * 60)

transforms = ants.registration(
    fixed=fixed,
    moving=moving,
    type_of_transform="SyNRA",
)

ants.image_write(transforms["warpedmovout"], "registered_Default.nii.gz")

# Calculate metrics for second registration
print("\nMetrics for default Registration:")
try:
    results_default = calculate_all_metrics(
        "output/sub-HC001_ses-01_T1w.nii.gz",
        "registered_Default.nii.gz",
        skip_ngf=False,
        skip_mind=False,
    )

    print(f"  MI:   {results_default['mi']:.6f}")
    print(f"  NMI:  {results_default['nmi']:.6f}")
    if results_default["ngf"] is not None:
        print(f"  NGF:  {results_default['ngf']:.6f}")
    if results_default["mind"] is not None:
        print(f"  MIND: {results_default['mind']:.6f}")

except Exception as e:
    print(f"Error calculating metrics: {e}")


# Second registration: Half iterations
print("\n" + "=" * 60)
print("REGISTRATION 2: Normaffine Half iterations (50, 30)")
print("=" * 60)

transforms = ants.registration(
    fixed=fixed,
    moving=moving,
    type_of_transform="SyNRA",
    reg_iterations=(50, 30),
)

ants.image_write(transforms["warpedmovout"], "registered_normaffine.nii.gz")

# Calculate metrics for second registration
print("\nMetrics for normaffine Registration:")
try:
    results_normaffine = calculate_all_metrics(
        "output/sub-HC001_ses-01_T1w.nii.gz",
        "registered_normaffine.nii.gz",
        skip_ngf=False,
        skip_mind=False,
    )

    print(f"  MI:   {results_normaffine['mi']:.6f}")
    print(f"  NMI:  {results_normaffine['nmi']:.6f}")
    if results_normaffine["ngf"] is not None:
        print(f"  NGF:  {results_normaffine['ngf']:.6f}")
    if results_normaffine["mind"] is not None:
        print(f"  MIND: {results_normaffine['mind']:.6f}")

except Exception as e:
    print(f"Error calculating metrics: {e}")


# First registration: Full iterations
print("=" * 60)
print("REGISTRATION 1: Full iterations (100, 50, 30)")
print("=" * 60)

transforms = ants.registration(
    fixed=fixed,
    moving=moving,
    type_of_transform="SyNRA",
    reg_iterations=(100, 50, 30),
    aff_iterations=(100,),
    aff_shrink_factors=(1,),
    aff_smoothing_sigmas=(0,),
)

ants.image_write(transforms["warpedmovout"], "registered_full.nii.gz")

# Calculate metrics for first registration
print("\nMetrics for Full Registration:")
try:
    results_full = calculate_all_metrics(
        "output/sub-HC001_ses-01_T1w.nii.gz",
        "registered_full.nii.gz",
        skip_ngf=False,
        skip_mind=False,
    )

    print(f"  MI:   {results_full['mi']:.6f}")
    print(f"  NMI:  {results_full['nmi']:.6f}")
    if results_full["ngf"] is not None:
        print(f"  NGF:  {results_full['ngf']:.6f}")
    if results_full["mind"] is not None:
        print(f"  MIND: {results_full['mind']:.6f}")

except Exception as e:
    print(f"Error calculating metrics: {e}")

# Second registration: Half iterations
print("\n" + "=" * 60)
print("REGISTRATION 2: Half iterations (50, 30)")
print("=" * 60)

transforms = ants.registration(
    fixed=fixed,
    moving=moving,
    type_of_transform="SyNRA",
    reg_iterations=(50, 30),
    aff_iterations=(100,),
    aff_shrink_factors=(1,),
    aff_smoothing_sigmas=(0,),
)

ants.image_write(transforms["warpedmovout"], "registered_half.nii.gz")

# Calculate metrics for second registration
print("\nMetrics for Half Registration:")
try:
    results_half = calculate_all_metrics(
        "output/sub-HC001_ses-01_T1w.nii.gz",
        "registered_half.nii.gz",
        skip_ngf=False,
        skip_mind=False,
    )

    print(f"  MI:   {results_half['mi']:.6f}")
    print(f"  NMI:  {results_half['nmi']:.6f}")
    if results_half["ngf"] is not None:
        print(f"  NGF:  {results_half['ngf']:.6f}")
    if results_half["mind"] is not None:
        print(f"  MIND: {results_half['mind']:.6f}")

except Exception as e:
    print(f"Error calculating metrics: {e}")

# Third registration: Single level
print("\n" + "=" * 60)
print("REGISTRATION 3: Single level (30)")
print("=" * 60)

transforms = ants.registration(
    fixed=fixed,
    moving=moving,
    type_of_transform="SyNRA",
    reg_iterations=(30,),
    aff_iterations=(100,),
    aff_shrink_factors=(1,),
    aff_smoothing_sigmas=(0,),
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

# Summary comparison
print("\n" + "=" * 60)
print("SUMMARY COMPARISON")
print("=" * 60)

try:
    print("Registration Method          |    MI    |   NMI    |   NGF    |  MIND")
    print("-" * 70)

    # Default registration
    mi_default = results_default["mi"] if "results_default" in locals() else 0
    nmi_default = results_default["nmi"] if "results_default" in locals() else 0
    ngf_default = (
        results_default["ngf"]
        if "results_default" in locals() and results_default["ngf"] is not None
        else 0
    )
    mind_default = (
        results_default["mind"]
        if "results_default" in locals() and results_default["mind"] is not None
        else 0
    )
    print(
        f"Default (50,30)             | {mi_default:8.4f} | {nmi_default:8.4f} | {ngf_default:8.4f} | {mind_default:8.4f}"
    )

    # Normaffine registration
    mi_normaffine = results_normaffine["mi"] if "results_normaffine" in locals() else 0
    nmi_normaffine = (
        results_normaffine["nmi"] if "results_normaffine" in locals() else 0
    )
    ngf_normaffine = (
        results_normaffine["ngf"]
        if "results_normaffine" in locals() and results_normaffine["ngf"] is not None
        else 0
    )
    mind_normaffine = (
        results_normaffine["mind"]
        if "results_normaffine" in locals() and results_normaffine["mind"] is not None
        else 0
    )
    print(
        f"Normaffine (50,30)          | {mi_normaffine:8.4f} | {nmi_normaffine:8.4f} | {ngf_normaffine:8.4f} | {mind_normaffine:8.4f}"
    )

    # Full registration
    mi_full = results_full["mi"] if "results_full" in locals() else 0
    nmi_full = results_full["nmi"] if "results_full" in locals() else 0
    ngf_full = (
        results_full["ngf"]
        if "results_full" in locals() and results_full["ngf"] is not None
        else 0
    )
    mind_full = (
        results_full["mind"]
        if "results_full" in locals() and results_full["mind"] is not None
        else 0
    )

    print(
        f"Full (100,50,30)            | {mi_full:8.4f} | {nmi_full:8.4f} | {ngf_full:8.4f} | {mind_full:8.4f}"
    )

    # Half registration
    mi_half = results_half["mi"] if "results_half" in locals() else 0
    nmi_half = results_half["nmi"] if "results_half" in locals() else 0
    ngf_half = (
        results_half["ngf"]
        if "results_half" in locals() and results_half["ngf"] is not None
        else 0
    )
    mind_half = (
        results_half["mind"]
        if "results_half" in locals() and results_half["mind"] is not None
        else 0
    )

    print(
        f"Half (50,30)                | {mi_half:8.4f} | {nmi_half:8.4f} | {ngf_half:8.4f} | {mind_half:8.4f}"
    )

    # Single level registration
    mi_last = results_last["mi"] if "results_last" in locals() else 0
    nmi_last = results_last["nmi"] if "results_last" in locals() else 0
    ngf_last = (
        results_last["ngf"]
        if "results_last" in locals() and results_last["ngf"] is not None
        else 0
    )
    mind_last = (
        results_last["mind"]
        if "results_last" in locals() and results_last["mind"] is not None
        else 0
    )

    print(
        f"Single (30)                 | {mi_last:8.4f} | {nmi_last:8.4f} | {ngf_last:8.4f} | {mind_last:8.4f}"
    )

    # Find best performing registration for each metric
    print("\nBest performing registration:")

    if (
        "results_full" in locals()
        and "results_half" in locals()
        and "results_last" in locals()
    ):
        # MI (higher is better)
        mi_values = [mi_full, mi_half, mi_last]
        mi_names = ["Full", "Half", "Single"]
        best_mi_idx = mi_values.index(max(mi_values))
        print(f"  Best MI:   {mi_names[best_mi_idx]} ({mi_values[best_mi_idx]:.6f})")

        # NMI (higher is better)
        nmi_values = [nmi_full, nmi_half, nmi_last]
        best_nmi_idx = nmi_values.index(max(nmi_values))
        print(f"  Best NMI:  {mi_names[best_nmi_idx]} ({nmi_values[best_nmi_idx]:.6f})")

        # NGF (more negative is better, so we look for minimum)
        if ngf_full != 0 and ngf_half != 0 and ngf_last != 0:
            ngf_values = [ngf_full, ngf_half, ngf_last]
            best_ngf_idx = ngf_values.index(min(ngf_values))
            print(
                f"  Best NGF:  {mi_names[best_ngf_idx]} ({ngf_values[best_ngf_idx]:.6f})"
            )

        # MIND (lower is better in loss terms, but this depends on implementation)
        if mind_full != 0 and mind_half != 0 and mind_last != 0:
            mind_values = [mind_full, mind_half, mind_last]
            best_mind_idx = mind_values.index(min(mind_values))
            print(
                f"  Best MIND: {mi_names[best_mind_idx]} ({mind_values[best_mind_idx]:.6f})"
            )

except Exception as e:
    print(f"Error in summary comparison: {e}")

print("\nRegistration comparison complete!")
