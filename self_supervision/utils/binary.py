import os

import numpy as np
from PIL import Image

BASE_PATH = "C:\\Users\\David\\Desktop\\only_tumors_kits"
MASK = "aggregated_MAJ_seg"
CLASS = 2


def create_dir(dir_name: str):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


# this script converts mask to 0 and 1 (1 means specific class)
if __name__ == '__main__':
    for case in os.listdir(BASE_PATH):
        print(f"CONVERTING TO BINARY MASK CASE {case}")

        for mask_file in sorted(os.listdir(f"{BASE_PATH}\\{case}\\{MASK}")):
            mask_np = np.array(Image.open(f"{BASE_PATH}\\{case}\\{MASK}\\{mask_file}"), dtype=np.uint8)
            mask_np = np.where(mask_np == CLASS, 1, 0)  # converting to 0 and 1
            Image.fromarray(mask_np).convert("L").resize((512, 512)).save(f"{BASE_PATH}\\{case}\\{MASK}\\{mask_file}")
