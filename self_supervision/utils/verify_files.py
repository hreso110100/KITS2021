import os
from copy import deepcopy

import numpy as np
from PIL import Image
from numpy import copy, stack, moveaxis

DATA_PATH = "C:\\Users\\David\\Desktop\\predictions_gan"
STORE_PATH = "C:\\Users\\David\\Desktop\\masked_pred"
CASE = "case_00120"
MASK = "aggregated_MAJ_seg"

# valid real masks
# if __name__ == '__main__':
#
#     for case_file in sorted(os.listdir(f"{DATA_PATH}")):
#         for mask_file in os.listdir(f"{DATA_PATH}\\{case_file}\\{MASK}"):
#             print(mask_file)
#             mask = Image.open(f"{DATA_PATH}\\{case_file}\\{MASK}\\{mask_file}")
#             mask_r = np.array(mask, dtype=np.uint8)
#             mask_g = deepcopy(mask_r)
#             mask_b = deepcopy(mask_r)
#
#             # add color to mask
#             for index, mask in enumerate([mask_r, mask_g, mask_b]):
#                 mask[mask == 1] = int(255 - ((index + 1) * 255))  # blue
#                 mask[mask == 2] = int(255 - ((index + 1) * 128))  # green tumor
#                 mask[mask == 3] = int(255 - ((index + 1) * 64))  # orange cyst
#
#             stacked = stack((mask_r, mask_g, mask_b))
#             stacked = moveaxis(stacked, 0, -1)  # must be channel last to open via PIL
#
#             # imaging_np = np.array(Image.open(f"{DATA_PATH}\\{CASE}\\imaging\\imaging_{mask_file.split('_')[1]}"),
#             #                       dtype=np.uint8)
#             # imaging_np = stack((imaging_np, copy(imaging_np), copy(imaging_np)))
#             # imaging_np = moveaxis(imaging_np, 0, -1)  # must be channel last to open via PIL
#             #
#             # blended = Image.blend(Image.fromarray(imaging_np), Image.fromarray(stacked), alpha=0.5)
#
#             if not os.path.exists(f"{STORE_PATH}\\{case_file}"):
#                 os.makedirs(f"{STORE_PATH}\\{case_file}")
#
#             Image.fromarray(stacked).save(f"{STORE_PATH}\\{case_file}\\{mask_file}")


if __name__ == '__main__':

    for mask_file in os.listdir(f"{DATA_PATH}"):
        print(mask_file)
        mask = Image.open(f"{DATA_PATH}\\{mask_file}")
        mask_r = np.array(mask, dtype=np.uint8)
        mask_g = deepcopy(mask_r)
        mask_b = deepcopy(mask_r)

        # add color to mask
        for index, mask in enumerate([mask_r, mask_g, mask_b]):
            mask[mask == 1] = int(255 - ((index + 1) * 255))  # blue
            mask[mask == 2] = int(255 - ((index + 1) * 128))  # green tumor
            mask[mask == 3] = int(255 - ((index + 1) * 64))  # orange cyst

        stacked = stack((mask_r, mask_g, mask_b))
        stacked = moveaxis(stacked, 0, -1)  # must be channel last to open via PIL

        if not os.path.exists(f"{STORE_PATH}\\{mask_file.split('_imaging'[0])}"):
            os.makedirs(f"{STORE_PATH}\\{mask_file.split('_imaging'[0])}")

        Image.fromarray(stacked).save(f"{STORE_PATH}\\{mask_file}")
