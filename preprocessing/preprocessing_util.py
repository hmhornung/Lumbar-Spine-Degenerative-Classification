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
    Resample scans for inference, with no coordinates available
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
    
    return image, new_spacing, new_coordintes

def load_sample(path, study, series):
    slices = [pydicom.read_file(path, study, series, scan) for scan in os.listdir(os.path.join(path, study, series))]
    
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
    
    coord_dict = { f'{row[0]}-{row[1]}' : [row[2], row[3], row[4]] for row in filtered_df}
    
    # fill in missing coords with -1
    for clo in condition_level_order:
        if clo not in coord_dict.keys():
            coord_dict[clo] = [0,0,0]
            
    coord_list = [coord_dict[clo] for clo in condition_level_order]
    
    #get mask for which conditions & levels are given for the series 1 / 0
    mask = [int(coord_dict[clo] != [0,0,0]) for clo in condition_level_order]
    
    return coord_list, mask

def load_label_data(path, study):
    '''
    Returns a list of one-hot encodings for each condition severity in a study (25 x 4)
    '''
    df = pd.read_csv(path)
    filtered_df = df[df['study_id'] == study]
    label_list = [ str(filtered_df[clo.lower().replace(' ', '_').replace('/', '_')].item()) for clo in condition_level_order]
    
    discrete_label_list = [ label_dict[label] for label in label_list ]
    
    one_hot_encoded_label_list = [ [0,0,0,0] for i in range(25) ]
    
    index = 0
    for label in discrete_label_list:
        one_hot_encoded_label_list[index][label] = 1
        index += 1
        
    return one_hot_encoded_label_list

# import os
# from preprocessing_util import load_coord_data

path = 'data/train.csv'
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