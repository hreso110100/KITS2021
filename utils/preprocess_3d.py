import os

import nibabel as nib

BASE_PATH = "C:\\Users\\David\\Desktop\\kits21\\imagesTr"
STORE_PATH = "C:\\Users\\David\\Desktop\\preprocessed_kits"


def create_dir(dir_name: str):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


if __name__ == '__main__':
    for case in os.listdir(BASE_PATH):
        print(f"### PREPROCESSING CASE: {case} ###")
        create_dir(f"{STORE_PATH}\\{case}")

        nifty = nib.load(f"{BASE_PATH}\\{case}")
        nifty_data = nib.load(f"{BASE_PATH}\\{case}").get_fdata()
        print(f"### Original shape: {nifty.shape} ###")

        slice_start = int(nifty.shape[0] / 2)
        slice_end = int(nifty.shape[0] / 2) + 3
        sliced_nifty = nifty.slicer[slice_start: slice_end, :, :]
        print(f"### New shape: {sliced_nifty.shape} ###")

        nib.save(sliced_nifty, f"{STORE_PATH}\\KITS21_{case.split('_')[1]}")
