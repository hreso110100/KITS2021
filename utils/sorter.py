import os
from distutils.dir_util import copy_tree

import numpy as np
from PIL import Image

BASE_PATH = "C:\\Users\\David\\Desktop\\data"
STORE_PATH = "C:\\Users\\David\\Desktop\\sorted_data"
MASK = "aggregated_MAJ_seg"

if __name__ == '__main__':

    for case in os.listdir(BASE_PATH):
        print(f"SORTING CASE {case}")

        class_3 = False
        class_2 = False

        for mask_file in sorted(os.listdir(f"{BASE_PATH}\\{case}\\{MASK}")):
            mask_np = np.array(Image.open(f"{BASE_PATH}\\{case}\\{MASK}\\{mask_file}"), dtype=np.uint8)
            if 3 in mask_np:
                class_3 = True
            elif 2 in mask_np and 3 not in mask_np:
                class_2 = True

        # copy imaging and specific mask to sort folder
        if class_3:
            copy_tree(f"{BASE_PATH}\\{case}\\{MASK}", f"{STORE_PATH}\\{3}\\{case}\\{MASK}")
            copy_tree(f"{BASE_PATH}\\{case}\\imaging", f"{STORE_PATH}\\{3}\\{case}\\imaging")
        elif class_2:
            copy_tree(f"{BASE_PATH}\\{case}\\{MASK}", f"{STORE_PATH}\\{2}\\{case}\\{MASK}")
            copy_tree(f"{BASE_PATH}\\{case}\\imaging", f"{STORE_PATH}\\{2}\\{case}\\imaging")
        else:
            copy_tree(f"{BASE_PATH}\\{case}\\{MASK}", f"{STORE_PATH}\\{1}\\{case}\\{MASK}")
            copy_tree(f"{BASE_PATH}\\{case}\\imaging", f"{STORE_PATH}\\{1}\\{case}\\imaging")