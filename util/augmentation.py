import numpy as np
import pandas as pd
import torch
import scipy.ndimage
from scipy.ndimage import rotate

#Normalization
def normalize_volume(volume):
    mean = np.mean(volume)
    std = np.std(volume)
    normalized_volume = (volume - mean) / std
    return normalized_volume

#random rotation
def random_rotation(volume, roi_coords, max_angle):
    """
    Perform random rotation on a 3D volume and adjust ROI coordinates accordingly.
    
    Parameters:
    - volume (numpy.ndarray): 3D volume to be rotated.
    - roi_coords (numpy.ndarray): Array of 3D coordinates from the original volume.
    - max_angle_range (tuple): Range of possible rotation angles (+/- max_angle) for each axis.
    
    Returns:
    - rotated_volume (numpy.ndarray): Rotated 3D volume.
    - rotated_roi_coords (numpy.ndarray): Transformed ROI coordinates in the rotated volume.
    """
    assert len(volume.shape) == 3, "ensure 3d volume input"
    
    # Generate random angles for rotation for each axis
    angles = np.random.uniform(-max_angle, max_angle, size=3)
    
    # Perform rotation around each axis using scipy's rotate
    rotated_volume = volume.copy()
    for axis, angle in enumerate(angles):
        rotated_volume = rotate(rotated_volume, angle, axes=(axis, (axis + 1) % 3), reshape=True)
    
    # Adjust ROI coordinates based on rotation and volume shape changes
    rotated_roi_coords = roi_coords.copy()
    for axis, angle in enumerate(angles):
        if angle != 0:
            rotated_roi_coords = rotate_coords(rotated_roi_coords, angle, axis, volume.shape, rotated_volume.shape)
    
    return rotated_volume, rotated_roi_coords

def rotate_coords(coords, angle, axis, original_shape, new_shape):
    """
    Rotate 3D coordinates around a specified axis by a given angle,
    taking into account volume shape changes after rotation.
    
    Parameters:
    - coords (numpy.ndarray): Array of 3D coordinates to be rotated.
    - angle (float): Angle of rotation in degrees.
    - axis (int): Axis around which rotation is performed (0, 1, or 2).
    - original_shape (tuple): Shape of the volume before rotation.
    - new_shape (tuple): Shape of the volume after rotation.
    
    Returns:
    - new_coords (numpy.ndarray): Transformed 3D coordinates after rotation.
    """
    assert axis in [0, 1, 2], "Axis must be 0, 1, or 2"
    
    print(f'{(axis, (axis + 1) % 3)}')
    # Create rotation matrix for the specified axis
    rotation_matrix = rotate(np.eye(3), angle, axes=(axis, (axis + 1) % 3), reshape=False)
    
    # Calculate center of the original volume
    center = np.array(original_shape) / 2
    
    # Translate coordinates so that center of original volume is at the origin
    coords_centered = coords - center
    
    # Apply rotation matrix to the centered coordinates
    rotated_coords_centered = np.dot(coords_centered, rotation_matrix.T)
    
    # Calculate new center after rotation
    new_center = np.array(new_shape) / 2
    
    # Translate coordinates so that center of rotated volume is at the new center
    new_coords = rotated_coords_centered + new_center
    
    return new_coords

#random scaling
def random_scaling(volume, roi_coords, max_scale=0.1):
    '''
    Performs random scaling on all 3 axes. Scale values are generated for each axis in specified range\n
    Generated by ChatGPT
    '''
    # Generate random scale factors for each axis within the range [-max_angle, +max_angle]
    scales = np.random.uniform(1 - max_scale, 1 + max_scale, size=3)
    scaled_volume = scipy.ndimage.zoom(volume, zoom=scales, order=1)

    def scale_coords(coords, scales, original_shape):
        return coords * scales# + (np.array(scaled_volume.shape) - np.array(original_shape) * scales) / 2

    scaled_roi_coords = scale_coords(roi_coords, scales, volume.shape)
    
    return scaled_volume, scaled_roi_coords

#gaussian noise
def gaussian_noise(volume, noise_std):
    noise = np.random.normal(0, noise_std, volume.shape)
    noisy_volume = volume + noise
    return noisy_volume

def random_shift_and_crop(volume, roi_coords, new_shape, max_shift):
    '''
    Crops / zero-pads volume into new dimensions, and applies random shifts to the volume relative to the center of the new shape\n
    Generated by ChatGPT
    '''
    assert len(new_shape) == 3, "new_shape must be a tuple of three dimensions (depth, height, width)"
    
    # Get current volume shape
    current_shape = volume.shape
    
    # Compute center indices
    current_center = np.array(current_shape) // 2
    new_center = np.array(new_shape) // 2
    
    # Generate random shifts within max_shift
    shifts = np.random.randint(-max_shift, max_shift + 1, size=3)
    
    # Calculate start and end indices for cropping
    start_idx = current_center - new_center + shifts
    
    # Initialize cropped volume and mask
    cropped_volume = np.zeros(new_shape)
    
    # Calculate valid crop range within current volume
    crop_start = np.maximum(-start_idx, 0)
    crop_end = np.minimum(current_shape - start_idx, new_shape)
    
    # Copy cropped region from original volume to cropped volume
    cropped_volume[crop_start[0]:crop_end[0],
                   crop_start[1]:crop_end[1],
                   crop_start[2]:crop_end[2]] = volume[start_idx[0]+crop_start[0]:start_idx[0]+crop_end[0],
                                                         start_idx[1]+crop_start[1]:start_idx[1]+crop_end[1],
                                                         start_idx[2]+crop_start[2]:start_idx[2]+crop_end[2]]
    
    # Adjust ROI coordinates based on cropping and shifts
    shifted_roi_coords = roi_coords - start_idx
    
    # Create mask for coordinates outside new shape
    outside_mask = (shifted_roi_coords[:, 0] < 0) | (shifted_roi_coords[:, 0] >= new_shape[0]) | \
                   (shifted_roi_coords[:, 1] < 0) | (shifted_roi_coords[:, 1] >= new_shape[1]) | \
                   (shifted_roi_coords[:, 2] < 0) | (shifted_roi_coords[:, 2] >= new_shape[2])
    outside_mask = [not idx for idx in outside_mask]
    outside_mask = np.array(outside_mask, dtype=np.int32)
    return cropped_volume, shifted_roi_coords, outside_mask