import os

import numpy as np
import yaml
from PIL import Image
from skimage import feature
from torch import tensor


class LoaderEdge:

    def __init__(self, shape: tuple):
        with open(f"C:\\Users\\David\\PycharmProjects\\KITS2021\\self_supervision\\config\\gan_config.yaml",
                  'r') as file:
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
            batch_edges = []

            # load case folder and select one frame
            case_folder = self.case_list[chosen_index]
            img_list = os.listdir(f"{self.dataset_folder}\\{case_folder}")
            img_list_index = np.random.choice(len(img_list), 1, replace=False)[0]

            try:
                loaded_img = np.asarray(
                    Image.open(f"{self.dataset_folder}\\{case_folder}\\{img_list[img_list_index]}"))
                edges = feature.canny(loaded_img).astype(np.uint8)
                loaded_edges = np.where(edges == 1, 255, edges)
                loaded_img = loaded_img + np.random.normal(
                    0, 1, (256, 256)).astype(np.uint8)

            except FileNotFoundError:
                print(f"LOGGER: Cannot load case {self.case_list[chosen_index]}.")
                continue

            # min max scaling of images to range -1 and 1
            loaded_img = ((loaded_img / 255.0) * 2) - 1
            loaded_edges = ((loaded_edges / 255.0) * 2) - 1

            batch_img.append(loaded_img.reshape(self.shape))
            batch_edges.append(loaded_edges.reshape(self.shape))

            yield np.array(batch_img), np.array(batch_edges)

    def save_data(self, epoch: int, edges: tensor, imaging: tensor, fake: tensor):
        """
        Saving data.
        :param epoch: Number of epochs.
        :param edges: Edges data.
        :param imaging: Imaging data.
        :param fake: Generated data.
        """

        self.save(epoch, edges, self.generated_folder, "edges")
        self.save(epoch, imaging, self.generated_folder, "imaging")
        self.save(epoch, fake, self.generated_folder, "generated")

    def save(self, epoch: int, data: tensor, folder: str, file_name: str):
        """
        Base method for saving.
        :param epoch: Number of epochs.
        :param data: Data to save.
        :param folder: Folder where data will be saved.
        :param file_name: Name of the final file. File will be PNG by default.
        """

        # rescaling back
        data = (((data + 1) / 2) * 255).astype(np.uint8)

        if not os.path.exists(f"{folder}/{str(epoch)}"):
            os.makedirs(f"{folder}/{str(epoch)}")

        Image.fromarray(data[:, :, 0]).save(f"{folder}/{str(epoch)}/{file_name}.png")
