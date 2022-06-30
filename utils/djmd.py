import numpy as np
from preprocess_jpeg import load_file

CONST_GAMA = 0.01

SHIFT_Y = 1024
SCALE_Y = 2040

def get_shift_scale_maxmin(dataset_x):
    SHIFT_VALUE_X = -np.amin(dataset_x)
    SCALE_VALUE_X = np.amax(dataset_x)
    SCALE_VALUE_X += SHIFT_VALUE_X
    
    return SHIFT_VALUE_X, SCALE_VALUE_X

def shift_and_normalize(batch, shift_value, scale_value):
    return ((batch+shift_value)/scale_value)+CONST_GAMA

def inv_shift_and_normalize(batch, shift_value, scale_value):
    return ((batch-CONST_GAMA)*scale_value)-shift_value

