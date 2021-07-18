import os
from distutils.dir_util import copy_tree

import numpy as np
from PIL import Image

BASE_PATH = "C:\\Users\\Admin\\Desktop\\data"
STORE_PATH = "C:\\Users\\Admin\\Desktop\\sorted_data"
MASK = "aggregated_MAJ_seg"


def create_dir(dir_name: str):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


if __name__ == '__main__':

    # create dirs for storing sorted data
    for cl_number in [1, 2, 3]:
        create_dir(f"{STORE_PATH}\\{cl_number}\\imaging")
        create_dir(f"{STORE_PATH}\\{cl_number}\\{MASK}")

    for case in os.listdir(BASE_PATH):
        dir_number = 0

        for mask_file in sorted(os.listdir(f"{BASE_PATH}\\{case}\\{MASK}")):
            mask_np = np.array(Image.open(f"{BASE_PATH}\\{case}\\{MASK}\\{mask_file}"), dtype=np.uint8)
            if 2 in mask_np and 3 in mask_np:
                dir_number = 3
            elif 2 in mask_np and 3 not in mask_np:
                dir_number = 2
            elif 2 not in mask_np and 3 in mask_np:
                dir_number = 3
            else:
                dir_number = 1

        # copy imaging and specific mask to sort folder
        copy_tree(f"{BASE_PATH}\\{case}\\{MASK}", f"{STORE_PATH}\\{dir_number}\\{MASK}")
        copy_tree(f"{BASE_PATH}\\{case}\\imaging", f"{STORE_PATH}\\{dir_number}\\imaging")
