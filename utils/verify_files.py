import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL.Image import Image

STORE_PATH = "C:\\Users\\Admin\\Desktop\\data"
CASE = "case_0000"
MASK = "aggregated_MAJ_seg"

if __name__ == '__main__':
    imaging = Image.open(f"{STORE_PATH}\\{CASE}\\imaging\\imaging.png")
    imaging_np = np.array(imaging.getdata())

    mask = Image.open(f"{STORE_PATH}\\{CASE}\\{MASK}\\mask_part90.png")
    mask_np = np.array(mask.getdata())

    color = np.array([0, 255, 0], dtype='uint8')

    masked_img = np.where(mask_np[..., None], color, imaging_np)
    out = cv2.addWeighted(imaging_np, 0.8, masked_img, 0.2, 0)

    plt.plot(out)
    plt.show()
