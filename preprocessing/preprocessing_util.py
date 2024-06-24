#imports
import numpy as np
import pandas as pd 
import pydicom
import os
import sys
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#path constants
ROOT_DIR = '../'
DATA_DIR = 'data/train_images/'
DEST_DIR = 'preprocessing/raw_data/'

def resample(image, scan, new_spacing=[1,1,1]):
    '''
    Resample scans for inference, with no coordinates available\n
    Based on the function from https://www.kaggle.com/code/gzuidhof/full-preprocessing-tutorial by GUIDO ZUIDHOF
    '''
    # Determine current pixel spacing
    spacing = np.array([scan[0].SpacingBetweenSlices] + list(scan[0].PixelSpacing), dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

def resample_with_coordinates(image, scan, coordinates_list: list, new_spacing=[1,1,1]):
    '''
    Resample scans for training, with coordinates available
    Based on the function from https://www.kaggle.com/code/gzuidhof/full-preprocessing-tutorial by GUIDO ZUIDHOF
    '''
    # Determine current pixel spacing
    spacing = np.array([scan[0].SpacingBetweenSlices] + list(scan[0].PixelSpacing), dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.zoom(image, real_resize_factor, mode='nearest')
    
    new_coordintes = [[coord_set[0] * real_resize_factor[0],
                       coord_set[1] * real_resize_factor[1],
                       coord_set[2] * real_resize_factor[2]] for coord_set in coordinates_list]
    
    return image, new_coordintes, new_spacing

def load_sample(path, study, series):
    slices = [pydicom.read_file(os.path.join(path, study, series, scan)) for scan in os.listdir(os.path.join(path, study, series))]
    
    return slices

def check_valid_scan(slices):
    '''
    Ensure Series has scans all of the same size \n
    Some series have multiple dicom scan sets dumped in them
    '''
    ref_dims = slices[0].pixel_array.shape
    for slice in slices:
        if slice.pixel_array.shape != ref_dims:
            return False
    return True

'''
Coordinate Prediciton Format
'''
condition_order = [
    'Spinal Canal Stenosis',
    'Right Neural Foraminal Narrowing',
    'Left Neural Foraminal Narrowing',
    'Right Subarticular Stenosis',
    'Left Subarticular Stenosis'
]
vertebrae_order = [
    'L1/L2',
    'L2/L3',
    'L3/L4',
    'L4/L5',
    'L5/S1',
]

condition_level_order = []

for condition in condition_order:
    for level in vertebrae_order:
        condition_level_order.append(f'{condition}_{level}')
        
label_dict = {'Normal/Mild': 0,
              'Moderate'   : 1,
              'Severe'     : 2,
              'nan'        : 3
             }  

def load_coord_data(path, series):
    '''
    Gets labeled coordinate data given for each series
    Returns a list of 3D coordinates for each condition-level & a mask for non-present data
    '''
    df = pd.read_csv(path)
    filtered_df = df[df['series_id'] == series][[ 'condition', 'level', 'instance_number', 'x', 'y']].values.tolist()
    
    # we want
    # 1) a list of 3-tuples of coordinates, corresponding to condition_level_order. [0,0,0] in the rest
    # 2) a mask corresponding to the present / missing predictions for the series 1 / 0
    
    coord_dict = { f'{row[0]}_{row[1]}' : [row[2], row[3], row[4]] for row in filtered_df}
    
    # fill in missing coords with -1
    for clo in condition_level_order:
        if clo not in coord_dict.keys():
            coord_dict[clo] = [0,0,0]
            
    coord_list = [coord_dict[clo] for clo in condition_level_order]
    
    #get mask for which conditions & levels are given for the series 1 / 0
    mask = [int(coord_dict[clo] != [0,0,0]) for clo in condition_level_order]
    
    return np.array(coord_list), np.array(mask)

def sagittal_to_axial(image : np.ndarray, coordinates):
    transformed_image = image.swapaxes(0,2)
    for i in range(len(coordinates)):
        temp = coordinates[i][0]
        coordinates[i][0] = coordinates[i][2]
        coordinates[i][2] = temp
        
    return transformed_image, coordinates

def load_label_data(path, study):
    '''
    Returns a list of one-hot encodings for each condition severity in a study (25 x 4)
    '''
    df = pd.read_csv(path)
    filtered_df = df[df['study_id'] == study]
    label_list = [ str(filtered_df[clo.lower().replace(' ', '_').replace('/', '_')].item()) if len(filtered_df[clo.lower().replace(' ', '_').replace('/', '_')]) > 0 else 'nan' for clo in condition_level_order]
    
    discrete_label_list = [ label_dict[label] for label in label_list ]
    
    one_hot_encoded_label_list = [ [0,0,0,0] for i in range(25) ]
    
    index = 0
    for label in discrete_label_list:
        one_hot_encoded_label_list[index][label] = 1
        index += 1
        
    return np.array(one_hot_encoded_label_list)

def get_series_orientation(path, series):
    df = pd.read_csv(path)
    print(df[df['series_id'] == series])
    description = df[df['series_id'] == series]['series_description'].item()
    return description

def crop_or_pad_image(image, target_dims = [400,400,400]):
    '''
    Crops and zero-pads dimensions larger / smaller than the target dimensions.\n
    dim_diffs has left / right difference per dimension. Negative means cropped, Positive means zero-padded
    '''
    dim_diffs = np.array([[0,0],[0,0],[0,0]])
    for i in range(3):
        size = image.shape[i]
        difference_left = (target_dims[i] - size) // 2
        difference_right = (target_dims[i] - size) - difference_left
        dim_diffs[i] = np.array([difference_left,difference_right])
        print(difference_left)
        print(difference_right)
        
    
    cropped_image = np.zeros(target_dims)
    
    crop_start = [dim_diffs[i,0] if dim_diffs[i,0] > 0 else 0 for i in range(3)]
    crop_end = [(target_dims[i] - dim_diffs[i,1]) if dim_diffs[i,1] > 0 else target_dims[i] for i in range(3)]
    
    image_start = [ 0 if dim_diffs[i,0] > 0 else abs(dim_diffs[i,0]) for i in range(3)]
    image_end = [ image.shape[i] if dim_diffs[i,0] > 0 else image.shape[i] + dim_diffs[i,1]  for i in range(3)]
    
    target_slices = tuple(slice(start, end) for start, end in zip(crop_start, crop_end))
    source_slices = tuple(slice(start, end) for start, end in zip(image_start, image_end))

    
    print(f'crop ranges:\nz:{crop_start[0]}-{crop_end[0]}\nz:{crop_start[1]}-{crop_end[1]}\ny:{crop_start[2]}-{crop_end[2]}')
    print(f'crop ranges:\nz:{image_start[0]}-{image_end[0]}\nz:{image_start[1]}-{image_end[1]}\ny:{image_start[2]}-{image_end[2]}')
    
    # cropped_image[slice(crop_start[0],crop_end[0]),slice(crop_start[1],crop_end[1]),slice(crop_start[2],crop_end[2])] = cropped_image[slice(image_start[0],image_end[0]),slice(image_start[1],image_end[1]),slice(image_start[2],image_end[2])]
    
    cropped_image[target_slices] = image[source_slices]
    
    return cropped_image, dim_diffs

def adjust_coordinates(coord_data, dim_adjustments, max_dims):
    assert coord_data.shape[0] == 25
    
    adjusted_coords = [coords + dim_adjustments for coords in coord_data]
    
    #create mask for coordinates that were cropped out of bounds
    mask = [int((coord > [0,0,0]).all() and (coord < max_dims).all()) for coord in adjusted_coords]
    mask = np.array(mask)
    
    #apply mask to remove OOB coordinates
    result = adjusted_coords * mask[:, np.newaxis]
    
    return result, mask



#crop testing
# example = np.random.rand(345, 775, 111)
# print(example.max())
# example, diffs = crop_or_pad_image(example, [755,229,105])

# print(example[1,0,1])
    
    
# import os
# from preprocessing_util import load_coord_data

# path = 'data/train.csv'
# 3846131846
# 2901066339
# 198404504

#10728036
# study = 97086905

# data = load_label_data(path, study)

# print(data)

#pipeline from raw data to batch training:
#
#   load_sample() -                     Stack dicom images & get coord list for them
#   resample(_with_coordinates)() -  resample to isomorphic resolution, transform coord list
#                   normalize pixel values
#                   transform sagittal to axial
#                   crop / zero-pad image in every dimension. Adjust coordinates. Record OOB coords
#                   Create the prediction mask
#