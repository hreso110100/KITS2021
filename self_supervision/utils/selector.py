import os
from shutil import copyfile

import numpy as np
from PIL import Image

BASE_PATH = "C:\\Users\\David\\Desktop\\sliced_kits"
STORE_PATH = "C:\\Users\\David\\Desktop\\only_kidneys_kits"
MASK = "aggregated_MAJ_seg"
CLASS = 1


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
            if CLASS in mask_np and 2 not in mask_np and 3 not in mask_np:
                # if CLASS in mask_np:
                image_np = np.array(Image.open(f"{BASE_PATH}\\{case}\\imaging\\imaging_{mask_file.split('_')[1]}"),
                                    dtype=np.uint8)
                Image.fromarray(image_np).convert("L").resize((256, 256)).save(
                    f"{STORE_PATH}\\{case}\\imaging\\imaging_{mask_file.split('_')[1]}")
                Image.fromarray(mask_np).convert("L").resize((256, 256)).save(
                    f"{STORE_PATH}\\{case}\\{MASK}\\{mask_file}")