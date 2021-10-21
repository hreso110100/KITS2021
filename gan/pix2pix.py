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
from gan.utils.loader import Loader


class GAN:

    def __init__(self, load_models=False, models_path=""):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        with open(f"../src/config/model_config.yml", 'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        self.samples_folder = self.config["folders"]["generated_images"]

        if not os.path.exists(self.samples_folder):
            os.makedirs(self.samples_folder)

        # default `log_dir` is "runs" - we'll be more specific here
        self.writer = SummaryWriter('../tensorboard')

        self.file_rows = 512
        self.file_cols = 512
        self.channels = 3
        self.file_shape = (self.channels, self.file_rows, self.file_cols)

        self.data_loader = Loader(shape=self.file_shape)

        self.losses = []
        self.coverages = []

        # Building losses
        self.loss_mse = MSELoss()
        self.loss_l1 = L1Loss()

        # Building discriminator
        self.d_patch = (1, int(self.file_rows // 2 ** 4), int(self.file_rows // 2 ** 4))

        # Choosing whether to load or create new models
        if load_models:
            self.discriminator, self.generator, self.optimizer_d, self.optimizer_g = self.load_models(path=models_path)
        else:
            self.discriminator = Discriminator(self.file_shape).to(self.device)
            # self.discriminator.apply(weights_init)
            self.optimizer_d = Adam(params=self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

            # Building generator
            self.generator = Generator(self.file_shape).to(self.device)
            # self.generator.apply(weights_init)
            self.optimizer_g = Adam(params=self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def prepare_sequences(self, batch_size=1, merged=False) -> tuple:
        """
        Preparing sequences of real and corrupted data.
        :return: Tuple of real and corrupted data.
        """

        real_data = []
        corrupted_data = []

        for _, (real, corrupted) in enumerate(self.data_loader.load_batch(batch_size)):
            real_data.append(real[0])
            corrupted_data.append(corrupted[0])

        return tensor(real_data, device=self.device).float(), tensor(corrupted_data, device=self.device).float()

    def sample_train_images(self, epoch, batch_size):
        """
        Continuous saving of data during training with coverage calculations.
        :param epoch: Current epoch.
        :param batch_size: Batch size.
        :return Coverage of users by drones.
        """

        real, corrupted = self.prepare_sequences(merged=True)
        fake = self.generator(corrupted)

        fake = fake.detach().cpu().numpy().swapaxes(1, 2).reshape(self.file_rows, self.channels)
        real = real.detach().cpu().numpy().swapaxes(1, 2).reshape(self.file_rows, self.channels)
        corrupted = corrupted.detach().cpu().numpy().swapaxes(1, 2).reshape(self.file_rows, self.channels)

        self.data_loader.save_data(epoch, batch_size, corrupted, real, fake)
        self.markers.create_map(data_list=[real, fake], epoch=epoch)

        return self.markers.calculate_coverage(data=[real, fake])

    def train(self, epochs: int, batch_size: int, sample_interval: int):
        start_time = datetime.datetime.now()

        # Adversarial ground truths
        valid = tensor(np.ones((batch_size,) + self.d_patch), requires_grad=False, device=self.device)
        fake = tensor(np.zeros((batch_size,) + self.d_patch), requires_grad=False, device=self.device)

        for epoch in range(epochs):
            real_A, real_B = self.prepare_sequences(batch_size, merged=True)
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

            self.losses.append({"D": loss_D, "G": loss_G})
            print(f"[Epoch {epoch}/{epochs}] [D loss: {loss_D}] [G loss: {loss_G}] time: {elapsed_time}")

            if epoch % sample_interval == 0:
                coverage = self.sample_train_images(epoch, batch_size)
                self.coverages.append(coverage)
                print(f"[Coverage: {coverage}]")

        # measure metrics
        self.plot_loss(self.losses)
        self.plot_coverage(self.coverages)

        self.generate_samples(100)
        self.save_models([self.discriminator, self.generator, self.optimizer_d, self.optimizer_g])

    def plot_loss(self, loss_list: list):
        """
        Plot losses of discriminator and generator.
        """

        plt.figure(figsize=(12, 5))
        loss_G = []
        loss_D = []

        for loss in loss_list:
            if self.device == "cuda":
                loss_G.append(loss["G"].cpu().detach().numpy())
                loss_D.append(loss["D"].cpu().detach().numpy())
            else:
                loss_G.append(loss["G"].detach().numpy())
                loss_D.append(loss["D"].detach().numpy())

        plt.plot(loss_G, label="Generator")
        plt.plot(loss_D, label="Discriminator")

        plt.title("Discriminator and generator loss")
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.show()

    def plot_coverage(self, coverage_list: list):
        """
        Plot coverage of drones.
        """

        _, bins, _ = plt.hist([round(x, 2) for x in coverage_list], bins=5, alpha=0.65, edgecolor='k')
        plt.axvline(np.array(coverage_list).mean(), color='k', linestyle='dashed', linewidth=1)
        plt.xticks(bins)
        plt.title("Pokrytie testovacích vzoriek")
        plt.xlabel("Pokrytie")
        plt.ylabel("Počet vzoriek")

        min_ylim, max_ylim = plt.ylim()
        plt.text(np.array(coverage_list).mean() * 1.005, max_ylim * 0.9,
                 'Priemer: {:.2f}'.format(np.array(coverage_list).mean()))

        plt.savefig(self.samples_folder)
        plt.show()

    def generate_samples(self, samples: int):
        """
        Generate N number of fake trajectories and store them on disc.
        :param samples: Number of samples to generate
        """
        for i in range(samples):
            print(f"LOGGER: Generating sample {i + 1}/{samples}.")
            real, corrupted = self.prepare_sequences(merged=True)
            fake = self.generator(corrupted)

            fake = fake.detach().cpu().numpy().swapaxes(1, 2).reshape(self.file_rows, self.channels)
            real = real.detach().cpu().numpy().swapaxes(1, 2).reshape(self.file_rows, self.channels)

            self.markers.create_map(data_list=[real, fake], epoch=i, save_location=self.samples_folder)

            coverage = self.markers.calculate_coverage([real, fake])
            self.coverages.append(coverage)

        self.plot_coverage(self.coverages)

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