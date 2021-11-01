import os
import random
import numpy as np
import yaml
from PIL import Image
from torch import tensor
from skimage import feature


class LoaderEdge:

    def __init__(self, shape: tuple):
        with open(f"C:\\Users\\David\\PycharmProjects\\KITS2021\\gan\\config\\model_config.yaml", 'r') as file:
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
                noise_tuple = list()
                for i in range(0, 3):
                    img = np.array(
                        Image.open(f"{self.dataset_folder}\\{case_folder}\\imaging\\{img_list[chosen_img + i]}"),
                        dtype=np.uint8)
                    img_tuple.append(img)
                    edges = feature.canny(img).astype(np.uint8)
                    noise_tuple.append(np.where(edges == 1, 255, edges))
            except FileNotFoundError:
                print(f"LOGGER: Cannot load case {self.case_list[chosen_index]}.")
                continue

            # min max scaling of images to range -1 and 1
            concatenated_img = ((np.dstack(img_tuple) / 255.0) * 2) - 1
            concatenated_noise = ((np.dstack(noise_tuple) / 255.0) * 2) - 1

            # adding noise
            concatenated_noise *= np.zeros((256, 256, 3))

            batch_img.append(concatenated_img.reshape(self.shape))
            batch_noise.append(concatenated_noise.reshape(self.shape))

            yield np.array(batch_img), np.array(batch_noise)

    def save_data(self, epoch: int, corrupted: tensor, real: tensor, fake: tensor):
        """
        Saving data.
        :param epoch: Number of epochs.
        :param corrupted: Corrupted data.
        :param real: Real data.
        :param fake: Generated data.
        """

        self.save(epoch, corrupted, self.generated_folder, "corrupted")
        self.save(epoch, real, self.generated_folder, "real")
        self.save(epoch, fake, self.generated_folder, "generated")

    def save(self, epoch: int, data: tensor, folder: str, file_name: str):
        """
        Base method for saving.
        :param epoch: Number of epochs.
        :param data: Data to save.
        :param folder: Folder where data will be saved.
        :param file_name: Name of the final file. File will be PNG by default.
        """

        data = (((data + 1) / 2) * 255).astype(np.uint8)

        # storing each channel as separate image
        if not os.path.exists(f"{folder}/{str(epoch)}"):
            os.makedirs(f"{folder}/{str(epoch)}")

        for index in range(data.shape[-1]):
            Image.fromarray(data[:, :, index]).save(f"{folder}/{str(epoch)}/{file_name}_{index}.png")
