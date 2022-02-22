import datetime
import os

import numpy as np
import torch
import yaml
from torch import tensor
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

# def weights_init(model):
#     """
#     Init weights for CNN layers.
#     :param model: Model to be initialized
#     """
#     classname = model.__class__.__name__
#     if classname.find("Conv") != -1:
#         torch.nn.init.xavier_uniform_(model.weight)
#         if model.bias is not None:
#             torch.nn.init.zeros_(model.bias)
from gan.discriminator import Discriminator
from gan.generator import Generator
from gan.utils.loader_edge import LoaderEdge


class GAN:

    def __init__(self, load_models=False, models_path=""):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        with open(f"C:\\Users\\David\\PycharmProjects\\KITS2021\\gan\\config\\model_config.yaml", 'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        self.samples_folder = self.config["folders"]["generated"]

        if not os.path.exists(self.samples_folder):
            os.makedirs(self.samples_folder)

        self.writer = SummaryWriter('../tensorboard')

        self.file_rows = 256
        self.file_cols = 256
        self.channels = 1
        self.file_shape = (self.channels, self.file_rows, self.file_cols)
        self.data_loader = LoaderEdge(shape=self.file_shape)

        # Building losses
        self.loss_mse = MSELoss()
        self.loss_l1 = L1Loss()

        # Building discriminator
        self.d_patch = (1, 16, 16)

        # Choosing whether to load or create new models
        if load_models:
            self.discriminator, self.generator, self.optimizer_d, self.optimizer_g = self.load_models(path=models_path)
        else:
            self.discriminator = Discriminator(self.file_shape).to(self.device)
            # self.discriminator.apply(weights_init)
            self.optimizer_d = Adam(params=self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

            # Building generator
            self.generator = Generator(self.file_shape).to(self.device)
            # self.generator.apply(weights_init)
            self.optimizer_g = Adam(params=self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    def train(self, epochs: int, batch_size: int, sample_interval: int):
        start_time = datetime.datetime.now()

        # Adversarial ground truths
        valid = tensor(np.ones((batch_size,) + self.d_patch), requires_grad=False, device=self.device)
        fake = tensor(np.zeros((batch_size,) + self.d_patch), requires_grad=False, device=self.device)

        for epoch in range(epochs):
            real_A, real_B = self.prepare_sequences(batch_size)
            fake_A = self.generator(real_B)

            #  Train Generator
            for param in self.discriminator.parameters():
                param.requires_grad_(False)

            self.optimizer_g.zero_grad()

            pred_fake = self.discriminator(fake_A, real_B)

            loss_mse = self.loss_mse(pred_fake, valid).double()
            loss_l1 = self.loss_l1(fake_A, real_A).double()

            # Total loss (100 is weight of L1 loss)
            loss_G = loss_mse + (100 * loss_l1)

            loss_G.backward()
            self.optimizer_g.step()

            #  Train Discriminator
            for param in self.discriminator.parameters():
                param.requires_grad_(True)

            self.optimizer_d.zero_grad()

            # Real loss
            pred_real = self.discriminator(real_A, real_B)
            loss_real = self.loss_mse(pred_real, valid)

            # Fake loss
            pred_fake = self.discriminator(fake_A.detach(), real_B)
            loss_fake = self.loss_mse(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            self.optimizer_d.step()

            elapsed_time = datetime.datetime.now() - start_time

            # measure losses
            print(f"[Epoch {epoch}/{epochs}] [D loss: {loss_D}] [G loss: {loss_G}] time: {elapsed_time}")
            self.writer.add_scalar('Training loss', loss_G, epoch)
            self.writer.add_scalar('Training loss', loss_D, epoch)

            if epoch % sample_interval == 0:
                self.sample_train_images(epoch)

        # self.generate_samples(10)
        self.save_models([self.discriminator, self.generator, self.optimizer_d, self.optimizer_g])

    def prepare_sequences(self, batch_size=1) -> tuple:
        """
        Preparing sequences of real and corrupted data.
        :return: Tuple of real and corrupted data.
        """

        imaging_data = []
        mask_data = []

        for _, (imaging, mask) in enumerate(self.data_loader.load_batch(batch_size)):
            imaging_data.append(imaging[0])
            mask_data.append(mask[0])

        return tensor(imaging_data, device=self.device).float(), tensor(mask_data, device=self.device).float()

    def sample_train_images(self, epoch):
        """
        Continuous saving of data during training with coverage calculations.
        :param epoch: Current epoch.
        :return Coverage of users by drones.
        """

        imaging, mask = self.prepare_sequences()
        fake = self.generator(mask)

        fake = fake.detach().cpu().numpy().reshape(self.file_rows, self.file_cols, self.channels)
        imaging = imaging.detach().cpu().numpy().reshape(self.file_rows, self.file_cols, self.channels)
        mask = mask.detach().cpu().numpy().reshape(self.file_rows, self.file_cols, self.channels)

        # visualize trained data and store them
        self.data_loader.save_data(epoch, mask, imaging, fake)

    # def generate_samples(self, samples: int):
    #     """
    #     Generate N number of fake trajectories and store them on disc.
    #     :param samples: Number of samples to generate
    #     """
    #     for i in range(samples):
    #         print(f"LOGGER: Generating sample {i + 1}/{samples}.")
    #         real, corrupted = self.prepare_sequences()
    #         fake = self.generator(corrupted)
    #
    #         fake = fake.detach().cpu().numpy().reshape(self.file_rows, self.file_cols, self.channels)
    #         real = real.detach().cpu().numpy().reshape(self.file_rows, self.file_cols, self.channels)
    #
    #         coverage = self.markers.calculate_coverage([real, fake])
    #         self.coverages.append(coverage)
    #
    #     self.plot_coverage(self.coverages)

    def save_models(self, models: list):
        """
        Saving trained models.
        :param models: List of pytorch models to be saved
        """
        formatted_datetime = (datetime.datetime.now()).strftime("%d_%m_%Y_%H_%M")
        save_path = f"../models/{formatted_datetime}"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for index, model in enumerate(models):
            torch.save(model.state_dict(), f"{save_path}/{index}_model_{model.__class__.__name__}.pth")

        print(f"LOGGER: Models successfully saved to {save_path}")

    def load_models(self, path: str):
        """
        Loading trained models.
        :param path: Path to folder where models are stored
        """

        discriminator, generator, optimizer_d, optimizer_g = [None, None, None, None]
        files = os.listdir(path)
        files.sort()

        for model in files:
            full_path = f"{path}/{model}"

            if "0" in model:
                discriminator = Discriminator(self.file_shape).to(self.device)
                discriminator.load_state_dict(torch.load(full_path))
            elif "1" in model:
                generator = Generator(self.file_shape).to(self.device)
                generator.load_state_dict(torch.load(full_path))
            elif "2" in model:
                optimizer_d = Adam(params=discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
                optimizer_d.load_state_dict(torch.load(full_path))
            elif "3" in model:
                optimizer_g = Adam(params=generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
                optimizer_g.load_state_dict(torch.load(full_path))

        print("LOGGER: Models successfully loaded.")
        return discriminator, generator, optimizer_d, optimizer_g


if __name__ == '__main__':
    model = GAN()
    model.train(10000, 8, 100)
