from PIL import Image
import os

import nibabel as nib
import numpy as np

BASE_PATH = "C:\\Users\\David\\PycharmProjects\\kits21\\kits21\\data"
STORE_PATH = "C:\\Users\\David\\Desktop\\sliced_kits"
MASKS_TO_CONVERT = ["aggregated_MAJ_seg.nii.gz"]  # here add name of the mask to convert


def create_dir(dir_name: str):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


if __name__ == '__main__':
    for case in os.listdir(BASE_PATH):
        print(f"### CONVERTING CASE: {case} ###")
        create_dir(f"{STORE_PATH}\\{case}\\imaging")

        # convert imaging to 2D
        imaging_data = nib.load(f"{BASE_PATH}\\{case}\\imaging.nii.gz").get_fdata()

        # convert mask files to 2D
        for file in MASKS_TO_CONVERT:
            data = nib.load(f"{BASE_PATH}\\{case}\\{file}").get_fdata()
            create_dir(f"{STORE_PATH}\\{case}\\{file.split('.')[0]}")

            # axial view
            for slice_n in range(data.shape[0]):
                img_slice = data[slice_n, :, :]
                # check if mask is empty, if yes ignore it
                if np.sum(img_slice) == 0:
                    continue
                Image.fromarray(img_slice).convert("L").resize((512, 512)).save(
                    f"{STORE_PATH}\\{case}\\{file.split('.')[0]}\\mask_part{slice_n}.png")

                imaging_slice = imaging_data[slice_n, :, :]
                Image.fromarray(imaging_slice).convert("L").resize((512, 512)).save(
                    f"{STORE_PATH}\\{case}\\imaging\\imaging_part{slice_n}.png")
