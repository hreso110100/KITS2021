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
    for case in os.listdir(BASE_PATH):
        dir_1 = False
        dir_2 = False
        dir_3 = False

        for mask_file in sorted(os.listdir(f"{BASE_PATH}\\{case}\\{MASK}")):
            mask_np = np.array(Image.open(f"{BASE_PATH}\\{case}\\{MASK}\\{mask_file}"), dtype=np.uint8)
            if 2 in mask_np and 3 in mask_np:
                dir_3 = True
            elif 2 in mask_np and 3 not in mask_np:
                dir_2 = True
            elif 2 not in mask_np and 3 in mask_np:
                dir_3 = True
            else:
                dir_1 = True

        # copy case to sort folder
        if dir_1:
            create_dir(f"{STORE_PATH}\\1")
            copy_tree(f"{BASE_PATH}\\{case}", f"{STORE_PATH}\\1")
        elif dir_2:
            create_dir(f"{STORE_PATH}\\2")
            copy_tree(f"{BASE_PATH}\\{case}", f"{STORE_PATH}\\2")
        else:
            create_dir(f"{STORE_PATH}\\3")
            copy_tree(f"{BASE_PATH}\\{case}", f"{STORE_PATH}\\3")
