import numpy as np
import pandas as pd
import torch
import scipy.ndimage
from scipy.ndimage import rotate
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
from scipy.ndimage import zoom

#Normalization
def normalize_volume(volume):
    mean = np.mean(volume)
    std = np.std(volume)
    normalized_volume = (volume - mean) / std
    return normalized_volume

def rotate_volume_and_coords(volume, coords, angle, axis):
    """
    Rotate a 3D volume and adjust ROI coordinates by a specified angle around a given axis.
    
    Parameters:
    - volume (numpy.ndarray): 3D volume to be rotated.
    - coords (numpy.ndarray): Array of 3D coordinates from the original volume.
    - angle (float): Angle of rotation in degrees.
    - axis (int): Axis around which rotation is performed (0, 1, or 2).
    
    Returns:
    - rotated_volume (numpy.ndarray): Rotated 3D volume.
    - rotated_coords (numpy.ndarray): Transformed ROI coordinates in the rotated volume.
    """
    assert axis in [0, 1, 2], "Axis must be 0 (X), 1 (Y), or 2 (Z)"
    
    # Perform rotation around the specified axis
    rotated_volume = rotate(volume, angle, axes=(axis, (axis + 1) % 3), reshape=True)
    
    # Calculate center of the original volume
    center = np.array(volume.shape) / 2
    
    # Translate coordinates so that center of original volume is at the origin
    coords_centered = coords - center
    
    # Create rotation matrix for the specified axis
    rotation_vector = np.zeros(3)
    rotation_vector[axis] = np.deg2rad(angle)
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
    
    # Apply rotation matrix to the centered coordinates
    rotated_coords_centered = np.dot(coords_centered, rotation_matrix.T)
    
    # Calculate new center after rotation
    new_center = np.array(rotated_volume.shape) / 2
    
    # Translate coordinates so that center of rotated volume is at the new center
    rotated_coords = rotated_coords_centered + new_center
    
    return rotated_volume, rotated_coords

def scale_3d_volume_and_coordinates(volume, scale_factors, coordinates):
    """
    Scales a 3D numpy array and a list of 3D coordinates by the given scale factors.

    Parameters:
    - volume: 3D numpy array to be scaled.
    - scale_factors: List of scale factors for each dimension.
    - coordinates: List of 3D coordinates to be scaled.

    Returns:
    - scaled_volume: Scaled 3D numpy array.
    - scaled_coordinates: List of scaled 3D coordinates.
    """
    # Scale the volume
    scaled_volume = zoom(volume, scale_factors, order=1)  # Order=1 for linear interpolation
    
    # Scale the coordinates
    scale_matrix = np.diag(scale_factors)
    scaled_coordinates = [np.dot(scale_matrix, coord) for coord in coordinates]

    return scaled_volume, np.array(scaled_coordinates)

#random scaling
def scale_volume(volume, roi_coords, max_scale):
    '''
    Performs random scaling on all 3 axes. Scale values are generated for each axis in specified range\n
    Generated by ChatGPT
    '''
    # Generate random scale factors for each axis within the range [-max_angle, +max_angle]
    scales = np.random.uniform(1 - max_scale, 1 + max_scale, size=3)
    scaled_volume = scipy.ndimage.zoom(volume, scales, order=1)
    # scaled_volume = F.interpolate(torch.from_numpy(volume).unsqueeze(0).unsqueeze(0), scale_factor=tuple(scales), mode='trilinear', align_corners=False)
    scaled_coords = roi_coords * scales
    
    return scaled_volume, scaled_coords, scales

#gaussian noise
def gaussian_noise(volume, max_noise_std):
    noise_std = np.random.rand() * max_noise_std
    noise = np.random.normal(0, noise_std, volume.shape)
    noisy_volume = volume + noise
    return noisy_volume

def random_shift_and_crop(volume, roi_coords, labels, old_mask, new_shape, max_shift):
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
    print(shifts)
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
    outside_mask = np.array([not idx for idx in outside_mask],dtype=np.int32)
    
    #combine masks
    final_mask = np.multiply(outside_mask, old_mask)
    
    #Mask coords that got cropped out
    cropped_coords = shifted_roi_coords * final_mask[:, np.newaxis]
    
    #Mask labels that got cropped out
    masked_labels = np.array([(0,0,0,1) if mask == 0 else labels[i] for i, mask in enumerate(final_mask)])
    
    return cropped_volume, cropped_coords, masked_labels,final_mask