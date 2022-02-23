import os

import numpy as np
from PIL import Image

BASE_PATH = "C:\\Users\\David\\Desktop\\only_tumors_kits"
STORE_PATH = "C:\\Users\\David\\Desktop\\only_tumors_cropped_kits"
MASK = "aggregated_MAJ_seg"
CLASS = 2


def create_dir(dir_name: str):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


if __name__ == '__main__':
    create_dir(STORE_PATH)

    for case in os.listdir(BASE_PATH):
        print(f"CROPPING CLASSES FOR CASE {case}")

        for mask_file in sorted(os.listdir(f"{BASE_PATH}\\{case}\\{MASK}")):
            mask_np = np.array(Image.open(f"{BASE_PATH}\\{case}\\{MASK}\\{mask_file}"), dtype=np.uint8)
            image_np = np.array(Image.open(f"{BASE_PATH}\\{case}\\imaging\\imaging_{mask_file.split('_')[1]}"),
                                dtype=np.uint8)

            result = mask_np * image_np
            Image.fromarray(result).convert("L").resize((512, 512)).save(f"{STORE_PATH}\\{case}_{mask_file.split('_')[1]}")
