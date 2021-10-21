import os
from shutil import copyfile

import numpy as np
from PIL import Image

BASE_PATH = "C:\\Users\\David\\Desktop\\sliced_kits"
STORE_PATH = "C:\\Users\\David\\Desktop\\only_tumors_kits"
MASK = "aggregated_MAJ_seg"
CLASS = 2


def create_dir(dir_name: str):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


if __name__ == '__main__':
    for case in os.listdir(BASE_PATH):
        print(f"SORTING CASE {case}")
        create_dir(f"{STORE_PATH}\\{case}\\imaging")
        create_dir(f"{STORE_PATH}\\{case}\\{MASK}")

        for mask_file in sorted(os.listdir(f"{BASE_PATH}\\{case}\\{MASK}")):
            mask_np = np.array(Image.open(f"{BASE_PATH}\\{case}\\{MASK}\\{mask_file}"), dtype=np.uint8)
            # if CLASS in mask_np and 2 not in mask_np and 3 not in mask_np:
            if CLASS in mask_np:
                copyfile(f"{BASE_PATH}\\{case}\\{MASK}\\{mask_file}",
                         f"{STORE_PATH}\\{case}\\{MASK}\\{mask_file}")
                copyfile(f"{BASE_PATH}\\{case}\\imaging\\imaging_{mask_file.split('_')[1]}",
                         f"{STORE_PATH}\\{case}\\imaging\\imaging_{mask_file.split('_')[1]}")
