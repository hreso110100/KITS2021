import datetime
import os

import numpy as np
import torch
import yaml
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks import one_hot
from monai.transforms import AsDiscrete
from torch import tensor
from torch.nn import Softmax
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from self_supervision.loaders.segmentation_loader import SegmentationLoader
from self_supervision.segmentation.unet import UNet


class Segmentation:

    def __init__(self, load_model=False, from_scratch=False, models_path=""):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        with open(f"C:\\Users\\David\PycharmProjects\\KITS2021\\self_supervision\\config\\model_config.yaml",
                  'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        self.dataset_train_folder = self.config["folders"]["dataset"]
        self.dataset_test_folder = self.config["folders"]["dataset_test"]
        self.dataset_val_folder = self.config["folders"]["dataset_validation"]

        self.writer = SummaryWriter('../tensorboard')

        self.input_rows = 256
        self.input_cols = 256
        self.input_channels = 1
        self.n_classes = 4
        self.file_shape = (self.input_channels, self.input_rows, self.input_cols)

        # Initialize loaders
        self.train_loader = SegmentationLoader(dataset_folder=self.dataset_train_folder, input_shape=self.file_shape,
                                               n_classes=self.n_classes)
        self.test_loader = SegmentationLoader(dataset_folder=self.dataset_test_folder,
                                              input_shape=self.file_shape, n_classes=self.n_classes)
        self.val_loader = SegmentationLoader(dataset_folder=self.dataset_val_folder,
                                             input_shape=self.file_shape, n_classes=self.n_classes)
        # validation modules
        self.softmax = Softmax(dim=1)
        self.threshold = AsDiscrete(threshold_values=True, logit_thresh=0.5)
        # Building loss
        self.loss_dice = DiceLoss(include_background=False, softmax=True, to_onehot_y=True)

        # Build metric
        self.dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")

        # Choosing whether to load or create new models
        if load_model:
            self.unet = self.load_model(path=models_path)
        else:
            # Building U-Net
            if from_scratch:
                # train U-Net from scratch
                self.unet = UNet(self.file_shape).to(self.device)
                self.optimizer = Adam(params=self.unet.parameters(), lr=0.0001, betas=(0.5, 0.999))
            else:
                # load weights from pretext task
                self.unet = self.load_model("C:\\Users\\David\\PycharmProjects\\KITS2021\\models\\24_02_2022_17_26")
                self.optimizer = Adam(params=self.unet.parameters(), lr=0.0001, betas=(0.5, 0.999))

    def train(self, epochs: int, batch_size: int, test_interval: int):
        start_time = datetime.datetime.now()

        for epoch in range(epochs):
            # set the model to training mode
            self.unet.train()
            average_batch_loss = 0

            # prepare batch and do prediction
            for (imaging, mask, _) in self.prepare_sequences(self.train_loader, batch_size):
                prediction = self.unet(imaging)
                # calculate loss and perform backprogation
                self.optimizer.zero_grad()
                train_loss = self.loss_dice(prediction, mask)
                average_batch_loss += train_loss.item()
                train_loss.backward()
                self.optimizer.step()

            # measure progress of the loss
            average_batch_loss = average_batch_loss / batch_size
            elapsed_time = datetime.datetime.now() - start_time
            print(f"LOGGER: [Epoch {epoch}/{epochs}] [U-Net loss: {average_batch_loss}] time: {elapsed_time}")
            self.writer.add_scalar('Training loss', average_batch_loss, epoch)

            # perform validation of training
            if epoch % test_interval == 0:
                # set the model to validation mode
                self.unet.eval()
                # switch off autograd
                with torch.no_grad():
                    # prepare batch and do prediction
                    for (imaging, mask, _) in self.prepare_sequences(self.train_loader, 8):
                        prediction = self.unet(imaging)
                        # calculate metric for each class
                        prediction = self.softmax(prediction)
                        prediction = self.threshold(prediction)

                        self.dice_metric_batch(y_pred=prediction, y=one_hot(mask, num_classes=4))

                    class_scores = self.dice_metric_batch.aggregate()
                    print(
                        f"LOGGER: DICE for class 1: {class_scores[0]}, class 2: {class_scores[1]},"
                        f" class 3: {class_scores[2]}")

                    # reset metric
                    self.dice_metric_batch.reset()

        with torch.no_grad():
            # set the model to validation mode
            self.unet.eval()

            # prepare batch and do prediction
            for (imaging, mask, prediction_case_name) in self.prepare_sequences(self.val_loader, 25):
                prediction = self.unet(imaging)

                # calculate metric for each class
                prediction = self.softmax(prediction)
                prediction = self.threshold(prediction)

                self.dice_metric_batch(y_pred=prediction, y=one_hot(mask, num_classes=4))

                # convert one hot encoded prediction back to 1-channel and save
                prediction = prediction.cpu().detach().numpy()
                for idx_channel, mask_channel in enumerate(prediction[0]):
                    prediction[0, idx_channel, :, :] = np.where(mask_channel == 1, idx_channel, 0)

                self.val_loader.save_data(prediction[0], prediction_case_name)

            class_scores = self.dice_metric_batch.aggregate()
            print(
                f"LOGGER: DICE for class 1: {class_scores[0]}, class 2: {class_scores[1]},"
                f" class 3: {class_scores[2]}")

            # reset metric
            self.dice_metric_batch.reset()

        self.save_model(self.unet)

    def prepare_sequences(self, data_loader: SegmentationLoader, batch_size=1) -> list:
        """
        Preparing sequences of real and mask data.
        :param batch_size: Size of the batch.
        :param data_loader: Instance of data loader class.
        :return: Tuple of real and mask data.
        """

        data = []

        for (imaging, mask, prediction_case_name) in data_loader.load_batch(batch_size):
            data.append((tensor(imaging, device=self.device, dtype=torch.float),
                         tensor(mask, device=self.device, dtype=torch.float), prediction_case_name))

        return data

    def save_model(self, unet: UNet):
        """
        Saving trained U-Net model.
        :param unet: U-Net model to be saved
        """
        formatted_datetime = (datetime.datetime.now()).strftime("%d_%m_%Y_%H_%M")
        save_path = f"C:\\Users\\David\\PycharmProjects\\KITS2021\\models\\segmentation\\{formatted_datetime}"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(unet.state_dict(), f"{save_path}/{model.__class__.__name__}.pth")

        print(f"LOGGER: U-Net model successfully saved to {save_path}")

    def load_model(self, path: str) -> UNet:
        """
        Loading pre-trained U-Net model.
        :param path: Path to folder where the model is stored
        """
        pretrained_weights = torch.load(f"{path}/1_model_Generator.pth")

        unet = UNet(self.file_shape).to(self.device)
        unet_weights = unet.state_dict()

        # filter out unnecessary weights
        pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in unet.state_dict()}

        # overwrite existing weights
        unet_weights.update(pretrained_weights)

        # load new weights
        unet.load_state_dict(unet_weights)

        # freeze encoder
        for index, param in enumerate(unet.parameters()):
            if index < 8:
                param.requires_grad = False

        print("LOGGER: Weights of U-Net successfully loaded.")
        return unet


if __name__ == '__main__':
    model = Segmentation(from_scratch=True)
    model.train(10000, 16, 100)
