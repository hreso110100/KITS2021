import logging
import os
import sys
import tempfile

import monai
import torch
from monai.data import list_data_collate, decollate_batch, ArrayDataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Activations, AsDiscrete, Compose, SaveImage, EnsureType, LoadImage, ScaleIntensity, \
    AddChannel
from torch.utils.data import DataLoader

DATA_PATH = "C:\\Users\\David\\Desktop\\data"


def create_data_pairs(mask_name: str, slice_start: int, slice_end: int):
    images = []
    seg = []

    for case in sorted(os.listdir(DATA_PATH))[slice_start:slice_end]:
        for mask in os.listdir(f"{DATA_PATH}\\{case}\\{mask_name}"):
            images.append(f"{DATA_PATH}\\{case}\\imaging\\imaging_{mask.split('_')[1]}")
            seg.append(f"{DATA_PATH}\\{case}\\{mask_name}\\{mask}")

    return images, seg


def validate():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # define transforms for image and segmentation
    imtrans = Compose([LoadImage(image_only=True), ScaleIntensity(), AddChannel(), EnsureType()])
    segtrans = Compose([LoadImage(image_only=True), AddChannel(), EnsureType()])

    images, segs = create_data_pairs("aggregated_MAJ_seg", 270, 300)
    val_ds = ArrayDataset(images, imtrans, segs, segtrans)
    # sliding window inference need to input 1 image in every iteration
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    saver = SaveImage(output_dir="./output", output_ext=".png", output_postfix="seg")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        dimensions=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    model.load_state_dict(torch.load("C:\\Users\\David\\PycharmProjects\\KITS2021\\network\\best_metric_model_segmentation2d_array.pth"))
    model.eval()

    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            # define sliding window size and batch size for windows inference
            roi_size = (128, 128)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            val_labels = decollate_batch(val_labels)
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)
            for val_output in val_outputs:
                saver(val_output)
        # aggregate the final mean dice result
        print("evaluation metric:", dice_metric.aggregate().item())
        # reset the status
        dice_metric.reset()


if __name__ == "__main__":
    validate()
