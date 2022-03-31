from PIL import Image
import os

import nibabel as nib
import numpy as np

BASE_PATH = "C:\\Users\\David\\Desktop\\ctk"
STORE_PATH = "C:\\Users\\David\\Desktop\\sliced_ctk"


def create_dir(dir_name: str):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


if __name__ == '__main__':
    for case in os.listdir(BASE_PATH):
        print(f"### CONVERTING CTK CASE: {case} ###")
        create_dir(f"{STORE_PATH}\\{case.split('.')[0]}")

        # convert imaging to 2D
        imaging_data = nib.load(f"{BASE_PATH}\\{case}").get_fdata()

        # convert mask files to 2D
        data = nib.load(f"{BASE_PATH}\\masks\\Case_{case.split('_')[1]}.nii.gz").get_fdata()

        # axial view
        for slice_n in range(data.shape[2]):
            img_slice = data[:, :, slice_n]
            # check if mask is empty, if yes ignore it
            if np.sum(img_slice) == 0:
                continue
            imaging_slice = imaging_data[:, :, slice_n]
            Image.fromarray(imaging_slice).rotate(90).convert("L").resize((256, 256), Image.LANCZOS).save(
                f"{STORE_PATH}\\{case.split('.')[0]}\\imaging_part{slice_n}.png")
