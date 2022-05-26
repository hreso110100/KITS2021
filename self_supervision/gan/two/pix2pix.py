import datetime
import os

import numpy as np
import torch
from torch import tensor
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam

from self_supervision.gan.two.discriminator import Discriminator
from self_supervision.gan.two.generator import Generator
from self_supervision.loaders.loader_edge import LoaderEdge


class GAN:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        self.discriminator = Discriminator(self.file_shape).to(self.device)
        self.optimizer_d = Adam(params=self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

        # Building generator
        self.generator = Generator(self.file_shape).to(self.device)
        self.optimizer_g = Adam(params=self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    def train(self, epochs: int, batch_size: int, sample_interval: int):
        start_time = datetime.datetime.now()

        # Adversarial ground truths
        valid = tensor(np.ones((batch_size,) + self.d_patch), requires_grad=False, device=self.device)
        fake = tensor(np.zeros((batch_size,) + self.d_patch), requires_grad=False, device=self.device)

        for epoch in range(epochs):
            img, edges = self.prepare_sequences(batch_size)
            fake_edges = self.generator(img)

            #  Train Generator
            for param in self.discriminator.parameters():
                param.requires_grad_(False)

            self.optimizer_g.zero_grad()

            pred_fake = self.discriminator(fake_edges, img)

            loss_mse = self.loss_mse(pred_fake, valid).double()
            loss_l1 = self.loss_l1(fake_edges, edges).double()

            # Total loss (100 is weight of L1 loss)
            loss_G = loss_mse + (100 * loss_l1)

            loss_G.backward()
            self.optimizer_g.step()

            #  Train Discriminator
            for param in self.discriminator.parameters():
                param.requires_grad_(True)

            self.optimizer_d.zero_grad()

            # Real loss
            pred_real = self.discriminator(edges, img)
            loss_real = self.loss_mse(pred_real, valid)

            # Fake loss
            pred_fake = self.discriminator(fake_edges.detach(), img)
            loss_fake = self.loss_mse(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            self.optimizer_d.step()

            elapsed_time = datetime.datetime.now() - start_time

            # measure losses
            print(f"LOGGER: [Epoch {epoch}/{epochs}] [D loss: {loss_D}] [G loss: {loss_G}] time: {elapsed_time}")

            if epoch % sample_interval == 0:
                self.sample_train_images(epoch)

        self.generate_samples(25)
        self.save_models([self.discriminator, self.generator, self.optimizer_d, self.optimizer_g])

    def prepare_sequences(self, batch_size=1) -> tuple:
        """
        Preparing sequences of real and corrupted data.
        :return: Tuple of real and corrupted data.
        """

        imaging_data = []
        mask_data = []

        for _, (imaging, edges) in enumerate(self.data_loader.load_batch(batch_size)):
            imaging_data.append(imaging[0])
            mask_data.append(edges[0])

        return tensor(imaging_data, device=self.device).float(), tensor(mask_data, device=self.device).float()

    def sample_train_images(self, epoch):
        """
        Continuous saving of data during training with coverage calculations.
        :param epoch: Current epoch.
        :return Coverage of users by drones.
        """

        imaging, edges = self.prepare_sequences()
        fake = self.generator(imaging)

        fake = fake.detach().cpu().numpy().reshape(self.file_rows, self.file_cols, self.channels)
        imaging = imaging.detach().cpu().numpy().reshape(self.file_rows, self.file_cols, self.channels)
        edges = edges.detach().cpu().numpy().reshape(self.file_rows, self.file_cols, self.channels)

        # visualize trained data and store them
        self.data_loader.save_data(epoch, edges, imaging, fake)

    def generate_samples(self, samples: int):
        """
        Generate N number of fake trajectories and store them on disc.
        :param samples: Number of samples to generate
        """
        for i in range(samples):
            print(f"LOGGER: Generating sample {i + 1}/{samples}.")
            imaging, edges = self.prepare_sequences()
            fake = self.generator(imaging)

            fake = fake.detach().cpu().numpy().reshape(self.file_rows, self.file_cols, self.channels)
            imaging = imaging.detach().cpu().numpy().reshape(self.file_rows, self.file_cols, self.channels)
            edges = edges.detach().cpu().numpy().reshape(self.file_rows, self.file_cols, self.channels)

            self.data_loader.save_data(i, edges, imaging, fake)

    def save_models(self, models: list):
        """
        Saving trained models.
        :param models: List of pytorch models to be saved
        """
        formatted_datetime = (datetime.datetime.now()).strftime("%d_%m_%Y_%H_%M")
        save_path = f"C:\\Users\\David\\PycharmProjects\\KITS2021\\models\\gan\\{formatted_datetime}"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for index, model in enumerate(models):
            torch.save(model.state_dict(), f"{save_path}/{index}_model_{model.__class__.__name__}.pth")

        print(f"LOGGER: Models successfully saved to {save_path}")

    def transform_weights(self, path: str):
        """
        Fetching pre-trained U-Net model weights from encoder part.
        :param path: Path to folder where the model is stored
        """
        pretrained_weights = torch.load(f"{path}/1_model_Generator.pth")

        pretrained_weights = {k: v for k, v in pretrained_weights.items() if 'transpose' not in k}

        for index, (k, v) in enumerate(pretrained_weights.items()):
            layer_idx = int(index / 4) + 1
            if 'down' in k:
                if 'Second' in k and 'bias' not in k:
                    v = torch.unsqueeze(v, 4)
                    v = torch.cat((v, v, v), dim=-1)
                    torch.save(v,
                               f"C:\\Users\\David\\PycharmProjects\\KITS2021\\models\\weights\\down_{layer_idx}_second.pt")
                elif 'Second' in k and 'bias' in k:
                    torch.save(v,
                               f"C:\\Users\\David\\PycharmProjects\\KITS2021\\models\\weights\\down_{layer_idx}_bias_second.pt")
                elif 'bias' in k:
                    torch.save(v,
                               f"C:\\Users\\David\\PycharmProjects\\KITS2021\\models\\weights\\down_{layer_idx}_bias_first.pt")
                else:
                    v = torch.unsqueeze(v, 4)
                    v = torch.cat((v, v, v), dim=-1)
                    torch.save(v,
                               f"C:\\Users\\David\\PycharmProjects\\KITS2021\\models\\weights\\down_{layer_idx}_first.pt")

            if 'up' and 'conv' in k:
                if '0' in k and 'bias' not in k:
                    v = torch.unsqueeze(v, 4)
                    v = torch.cat((v, v, v), dim=-1)
                    torch.save(v,
                               f"C:\\Users\\David\\PycharmProjects\\KITS2021\\models\\weights\\up_{layer_idx - 7}_first.pt")
                elif '0' in k and 'bias' in k:
                    torch.save(v,
                               f"C:\\Users\\David\\PycharmProjects\\KITS2021\\models\\weights\\up_{layer_idx - 7}_bias_first.pt")
                elif 'bias' in k:
                    torch.save(v,
                               f"C:\\Users\\David\\PycharmProjects\\KITS2021\\models\\weights\\up_{layer_idx - 7}_bias_second.pt")
                else:
                    v = torch.unsqueeze(v, 4)
                    v = torch.cat((v, v, v), dim=-1)
                    torch.save(v,
                               f"C:\\Users\\David\\PycharmProjects\\KITS2021\\models\\weights\\up_{layer_idx - 7}_second.pt")


if __name__ == '__main__':
    model = GAN()
    model.train(20000, 4, 100)
    # model.transform_weights("C:\\Users\\David\\PycharmProjects\\KITS2021\\models\\gan\\08_05_2022_18_19")
