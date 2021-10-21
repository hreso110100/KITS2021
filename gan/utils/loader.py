import os
import random

import numpy as np
import yaml
from PIL import Image
from torch import tensor


class Loader:

    def __init__(self, shape: tuple):
        with open(f"../config/model_config.yaml", 'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        self.dataset_folder = self.config["folders"]["dataset"]
        self.generated_folder = self.config["folders"]["generated"]

        if not os.path.exists(self.generated_folder):
            os.makedirs(self.generated_folder)

        self.case_list = os.listdir(self.dataset_folder)
        self.shape = shape

    def load_batch(self, batch_size=1):
        """
        Load data from given folder.
        :return: Tuple containing original data and corrupted data.
        """

        # Selecting random cases
        chosen_cases = np.random.choice(len(self.case_list), batch_size, replace=False)
        for _, chosen_index in enumerate(chosen_cases):
            batch_img = []
            batch_noise = []
            case_folder = self.case_list[chosen_index]
            img_list = os.listdir(f"{self.dataset_folder}\\{case_folder}\\imaging")
            # this represent index of the first image to be concatenated
            chosen_img = random.randint(0, len(img_list) - 3)

            try:
                img_tuple = list()
                mask_tuple = list()
                for i in range(0, 3):
                    img_tuple.append(np.array(
                        Image.open(f"{self.dataset_folder}\\{case_folder}\\imaging\\{img_list[chosen_img + i]}"),
                        dtype=np.uint8))
                    mask_tuple.append(np.array(
                        Image.open(
                            f"{self.dataset_folder}\\{case_folder}\\aggregated_MAJ_seg\\mask_{img_list[chosen_img + i].split('_')[1]}"),
                        dtype=np.uint8))
            except FileNotFoundError:
                print(f"LOGGER: Cannot load case {self.case_list[chosen_index]}.")
                continue

            # min max scaling of images
            concatenated_img = np.dstack(img_tuple) / 255.0
            concatenated_mask = np.dstack(mask_tuple) / 255.0

            noise_mask = np.random.choice([0, 1], size=(512, 512, 3), p=[.9, .1]) * concatenated_mask

            batch_img.append(concatenated_img.reshape(self.shape))
            batch_noise.append(noise_mask.reshape(self.shape))

            yield np.array(batch_img), np.array(batch_noise)

    def save_data(self, epoch: int, batch: int, corrupted: tensor, real: tensor, fake: tensor):
        """
        Saving data.
        :param epoch: Number of epochs.
        :param batch: Size of batch.
        :param corrupted: Corrupted data.
        :param real: Real data.
        :param fake: Generated data.
        """

        self.save(epoch, batch, corrupted, self.generated_folder, "corrupted")
        self.save(epoch, batch, real, self.generated_folder, "real")
        self.save(epoch, batch, fake, self.generated_folder, "generated")

    def save(self, epoch: int, batch: int, data: tensor, folder: str, file_name: str):
        """
        Base method for saving.
        :param epoch: Number of epochs.
        :param batch: Number of batches.
        :param data: Data to save.
        :param folder: Folder where data will be saved.
        :param file_name: Name of the final file. File will be PNG by default.
        """

        data = (data * 255).astype(np.uint8)

        # storing each channel as separate image
        for index in range(data.shape[-1]):
            Image.fromarray(data[:, :, index]).save(f"{folder}\\{str(epoch)}_{str(batch)}\\{file_name}_{index}.png")
