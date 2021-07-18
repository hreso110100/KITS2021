import logging
import os
import sys

import monai
import torch
from monai.data import decollate_batch, ArrayDataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImage,
    ScaleIntensity,
    EnsureType, AddChannel, )
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


def train():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # define transforms for image and segmentation
    train_imtrans = Compose([LoadImage(image_only=True), ScaleIntensity(), AddChannel(), EnsureType()])
    train_segtrans = Compose(
        [LoadImage(image_only=True), EnsureType(), AddChannel(), AsDiscrete(to_onehot=True, n_classes=4)])

    test_imtrans = Compose([LoadImage(image_only=True), ScaleIntensity(), AddChannel(), EnsureType()])
    test_segtrans = Compose(
        [LoadImage(image_only=True), EnsureType(), AddChannel(), AsDiscrete(to_onehot=True, n_classes=4)])

    # create a training data loader
    images, segs = create_data_pairs("aggregated_MAJ_seg", 0, 210)
    train_ds = ArrayDataset(images, train_imtrans, segs, train_segtrans)
    train_loader = DataLoader(train_ds, batch_size=8, num_workers=8, pin_memory=torch.cuda.is_available())
    im, seg = monai.utils.misc.first(train_loader)
    print(im.shape, seg.shape)
    # create a validation data loader
    images, segs = create_data_pairs("aggregated_MAJ_seg", 210, 270)
    test_ds = ArrayDataset(images, test_imtrans, segs, test_segtrans)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())

    dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    post_trans = Compose([EnsureType(), Activations(softmax=True), AsDiscrete(threshold_values=True)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        dimensions=2,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = monai.losses.DiceLoss(include_background=False, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, amsgrad=True)

    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_overall_values = list()
    metric_values_class_1 = list()
    metric_values_class_2 = list()
    metric_values_class_3 = list()
    writer = SummaryWriter()

    for epoch in range(50):
        print("-" * 20)
        print(f"epoch {epoch + 1}/{100}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval != 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in test_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    val_outputs = model(val_images)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # calculate metric for each class
                metric_class = dice_metric_batch.aggregate()
                metric_values_class_1.append(metric_class[0].item())
                metric_values_class_2.append(metric_class[1].item())
                metric_values_class_3.append(metric_class[2].item())

                # calculate overall metric
                metric_overall = dice_metric.aggregate().item()
                metric_overall_values.append(metric_overall)

                writer.add_scalar("val_mean_dice", metric_overall, epoch + 1)
                writer.add_scalar("val_dice_class_1", metric_class[0].item(), epoch + 1)
                writer.add_scalar("val_dice_class_2", metric_class[1].item(), epoch + 1)
                writer.add_scalar("val_dice_class_3", metric_class[2].item(), epoch + 1)

                if metric_overall > best_metric:
                    best_metric = metric_overall
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric_overall, best_metric, best_metric_epoch
                    )
                )
                print(
                    f"DICE FOR class 1: {metric_class[0].item()}, class 2: {metric_class[1].item()},"
                    f" class 3: {metric_class[2].item()}")
                # reset metrics
                dice_metric_batch.reset()
                dice_metric.reset()

                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label", max_channels=4)
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output", max_channels=4)

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    train()
