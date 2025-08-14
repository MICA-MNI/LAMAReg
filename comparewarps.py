import nibabel as nib
import ants
import numpy as np

def combine_warps_and_transform(
    first_warp_path, 
    second_warp_path,
    combined_warp_output_path,
    fixed_image_path,
    moving_image_path,
    affine_path,
    warped_output_path,
    interpolator="linear"
):
    """
    Combines two warp fields and applies them with an affine transform to register an image.
    
    Args:
        first_warp_path (str): Path to the first warp field
        second_warp_path (str): Path to the second warp field
        combined_warp_output_path (str): Path to save the combined warp field
        fixed_image_path (str): Path to the reference/fixed image
        moving_image_path (str): Path to the moving image to be transformed
        affine_path (str): Path to the affine transformation file
        warped_output_path (str): Path to save the warped output image
        interpolator (str): Interpolation method (default: "linear")
        
    Returns:
        ANTsImage or None: The warped output image if successful, None otherwise
    """
    # Load and combine warp fields
    second_warp = nib.load(second_warp_path)
    first_warp = nib.load(first_warp_path)
    
    second_arr = second_warp.get_fdata().squeeze()
    first_arr = first_warp.get_fdata().squeeze()
    
    combined_arr = first_arr + second_arr
    combined_warp = nib.Nifti1Image(combined_arr, first_warp.affine, first_warp.header)
    combined_warp.to_filename(combined_warp_output_path)
    
    # Load images
    fixed_img = ants.image_read(fixed_image_path)
    moving_img = ants.image_read(moving_image_path)
    
    # Apply transforms
    output = ants.apply_transforms(
        fixed=fixed_img,
        moving=moving_img,
        transformlist=[combined_warp_output_path, affine_path],
        interpolator=interpolator
    )
    
    if output is None:
        print("Error: Transform application failed!")
        return None
    else:
        output.to_filename(warped_output_path)
        print('Transform applied successfully')
        return output

# Example usage:
if __name__ == "__main__":
    result = combine_warps_and_transform(
        first_warp_path="/host/verges/tank/data/ian/LAMAReg-Experiments/flair_to_t1w_warp_stage2.nii.gz",
        second_warp_path="/host/verges/tank/data/ian/LAMAReg-Experiments/flair_to_t1w_warp_stage2_2.nii.gz",
        combined_warp_output_path="/host/verges/tank/data/ian/LAMAReg-Experiments/flair_to_t1w_warp_stage2_combined.nii.gz",
        fixed_image_path="example_data/sub-HC001_ses-01_T1w.nii.gz",
        moving_image_path="example_data/sub-HC001_ses-02_space-dwi_desc-b0.nii.gz",
        affine_path="/host/verges/tank/data/ian/LAMAReg-Experiments/flair_to_t1w_affine.mat",
        warped_output_path="example_data/sub-HC001_ses-02_space-dwi_desc-b0_warped.nii.gz"
    )