import logging
import os
import sys

import monai
import torch
from monai.data import decollate_batch, ArrayDataset
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Activations, AsDiscrete, Compose, SaveImage, EnsureType, LoadImage, ScaleIntensity, \
    AddChannel
from monai.visualize import plot_2d_or_3d_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
    writer = SummaryWriter()

    # define transforms for image and segmentation
    imtrans = Compose([LoadImage(image_only=True), ScaleIntensity(), AddChannel(), EnsureType()])
    segtrans = Compose([LoadImage(image_only=True), AddChannel(), EnsureType()])

    images, segs = create_data_pairs("aggregated_MAJ_seg", 270, 300)
    val_ds = ArrayDataset(images, imtrans, segs, segtrans)
    # sliding window inference need to input 1 image in every iteration
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    post_trans = Compose([EnsureType(), Activations(softmax=True), AsDiscrete(threshold_values=True, logit_thresh=0.6)])
    saver = SaveImage(output_dir="./output", output_ext=".png", output_postfix="seg")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        dimensions=2,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=4,
    ).to(device)
    model.load_state_dict(
        torch.load("C:\\Users\\David\\PycharmProjects\\KITS2021\\network\\best_metric_model_segmentation2d_array.pth"))
    model.eval()

    with torch.no_grad():
        for step, val_data in enumerate(val_loader):
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            val_outputs = model(val_images)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

            # compute metric for current iteration
            dice_metric_batch(y_pred=val_outputs, y=val_labels)
            metric_class = dice_metric_batch.aggregate()
            for val_output in val_outputs:
                saver(val_output)

            # store metric values to tensorboard
            writer.add_scalar("val_dice_class_1", metric_class[0].item(), step + 1)
            writer.add_scalar("val_dice_class_2", metric_class[1].item(), step + 1)
            writer.add_scalar("val_dice_class_3", metric_class[2].item(), step + 1)
            writer.add_scalar("val_dice_class_4", metric_class[3].item(), step + 1)

            dice_metric_batch.reset()

            # plot the last model output as GIF image in TensorBoard with the corresponding image and label
            plot_2d_or_3d_image(val_images, step + 1, writer, index=0, tag="image")
            plot_2d_or_3d_image(val_labels, step + 1, writer, index=0, tag="label", max_channels=4)
            plot_2d_or_3d_image(val_outputs, step + 1, writer, index=0, tag="output", max_channels=4)


if __name__ == "__main__":
    validate()
