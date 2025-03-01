import numpy as np
from PIL import Image

img = Image.open('/home/saadbk/SBK/Capstone/Files/optic_disk_mask.png')

img_arr = np.array(img) > 128

np.save('/home/saadbk/SBK/Capstone/Files/optic_disk_mask.npy',img_arr)