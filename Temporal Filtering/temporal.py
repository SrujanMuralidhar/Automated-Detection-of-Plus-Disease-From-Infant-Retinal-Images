from inference import get_mask
from PIL import Image
from scipy.ndimage import label
import numpy as np

def check_temporal(img):
    mask = get_mask(img)
    labeled_array, num_features = label(mask)
    # print(num_features)
    
    if(num_features == 1):
        mask_arr = np.array(mask) > 128
        temporal_arr = np.load('Files/optic_disk_mask.npy')
        result_img = np.logical_or(temporal_arr, mask_arr)
        if(np.array_equal(result_img,temporal_arr)):
            return False
        else:
            temporal_arr_left = np.load('Files/optic_disk_mask_left.npy')
            result_img = np.logical_or(temporal_arr_left,mask_arr)
            if(np.array_equal(result_img,temporal_arr_left)):
                return False
            else:
                return True
    else:
        return False

if(__name__ == "__main__"):
    path = '/home/saadbk/SBK/Capstone/FHPL_83-2022_20220922130851_20221020114517_Image_12_0008_2022-10-20_11-47-38-337_1526.png'
    check_temporal(path)