from copy import deepcopy

import numpy as np
from PIL import Image
from numpy import copy, stack, moveaxis

STORE_PATH = "C:\\Users\\Admin\\Desktop\\data"
CASE = "case_00000"
MASK = "aggregated_MAJ_seg"

if __name__ == '__main__':
    imaging = Image.open(f"{STORE_PATH}\\{CASE}\\imaging\\imaging_part236.png")
    imaging_np = np.array(imaging, dtype=np.uint8)
    imaging_np = stack((imaging_np, copy(imaging_np), copy(imaging_np)))
    imaging_np = moveaxis(imaging_np, 0, -1)  # must be channel last to open via PIL

    mask = Image.open(f"{STORE_PATH}\\{CASE}\\{MASK}\\mask_part236.png")
    mask_r = np.array(mask, dtype=np.uint8)
    mask_g = deepcopy(mask_r)
    mask_b = deepcopy(mask_r)

    # add color to mask
    for index, mask in enumerate([mask_r, mask_g, mask_b]):
        mask[mask == 1] = int(255 - ((index + 1) * 255))  # blue
        mask[mask == 2] = int(255 - ((index + 1) * 128))  # green
        mask[mask == 3] = int(255 - ((index + 1) * 64))  # orange

    stacked = stack((mask_r, mask_g, mask_b))
    stacked = moveaxis(stacked, 0, -1)  # must be channel last to open via PIL

    blended = Image.blend(Image.fromarray(imaging_np), Image.fromarray(stacked), alpha=0.5)
    Image.show(blended)
