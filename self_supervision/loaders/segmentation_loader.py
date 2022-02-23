import os

import numpy as np
import yaml
from PIL import Image
from torch import tensor


class SegmentationLoader:

    def __init__(self, dataset_folder: str, input_shape: tuple, n_classes: int):
        with open(f"C:\\Users\\David\\PycharmProjects\\KITS2021\\self_supervision\\config\\model_config.yaml",
                  'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        self.dataset_folder = dataset_folder
        self.predictions_folder = self.config["folders"]["predictions"]

        if not os.path.exists(self.predictions_folder):
            os.makedirs(self.predictions_folder)

        self.case_list = os.listdir(self.dataset_folder)
        self.input_shape = input_shape
        self.n_classes = n_classes

    def load_batch(self, batch_size=1):
        """
        Load data from given folder.
        :return: Tuple containing imagings and masks.
        """

        # Selecting random cases
        chosen_cases = np.random.choice(len(self.case_list), batch_size, replace=False)
        for _, chosen_index in enumerate(chosen_cases):
            batch_img = []
            batch_mask = []

            # load case folder and select one slice
            case_folder = self.case_list[chosen_index]
            img_list = os.listdir(f"{self.dataset_folder}\\{case_folder}\\imaging")
            img_list_index = np.random.choice(len(img_list), 1, replace=False)[0]

            try:
                loaded_img, loaded_mask = self.load_files(case_folder, img_list[img_list_index])

                # we need to check if mask is valid, some wierd bug occurs in some of them :(
                if np.max(loaded_mask) > 3:
                    print(
                        f"LOGGER: Invalid mask found at case {case_folder}:{img_list[img_list_index]}. Generating new batch file.")
                    img_list_index = np.random.choice(len(img_list), 1, replace=False)[0]
                    loaded_img, loaded_mask = self.load_files(case_folder, img_list[img_list_index])
            except FileNotFoundError:
                print(f"LOGGER: Cannot load case {self.case_list[chosen_index]}.")
                continue

            # min max scaling of imaging
            loaded_img = loaded_img / 255.0

            batch_img.append(loaded_img.reshape(self.input_shape))
            batch_mask.append(loaded_mask)

            yield np.array(batch_img), np.array(batch_mask)

    def load_files(self, case_folder, img_path):
        """
        Load imaging with corresponding mask.

        :param case_folder: Folder with case.
        :param img_path: Path to file.
        :return: Tuple of imaging and mask.
        """
        loaded_img = np.asarray(
            Image.open(f"{self.dataset_folder}\\{case_folder}\\imaging\\{img_path}"))
        loaded_mask = np.asarray(
            Image.open(
                f"{self.dataset_folder}\\{case_folder}\\aggregated_MAJ_seg\\mask_{img_path.split('_')[1]}"))

        return loaded_img, loaded_mask

    def save_data(self, epoch: int, mask: tensor, imaging: tensor, prediction: tensor):
        """
        Saving data.
        :param epoch: Number of epochs.
        :param mask: Mask data.
        :param imaging: Imaging data.
        :param prediction: Prediction data.
        """

        self.save(epoch, mask, self.predictions_folder, "mask")
        self.save(epoch, imaging, self.predictions_folder, "imaging")
        self.save(epoch, prediction, self.predictions_folder, "prediction")

    def save(self, epoch: int, data: tensor, folder: str, file_name: str):
        """
        Base method for saving.
        :param epoch: Number of epochs.
        :param data: Data to save.
        :param folder: Folder where data will be saved.
        :param file_name: Name of the final file. File will be PNG by default.
        """

        if file_name == "prediction":
            # TODO tu bude treba asi nejaky check iny
            data = (data * 3).astype(np.uint8)
        elif file_name == "imaging":
            data = (data * 255).astype(np.uint8)

        # storing each channel as separate image
        if not os.path.exists(f"{folder}/{str(epoch)}"):
            os.makedirs(f"{folder}/{str(epoch)}")

        Image.fromarray(data[:, :, 0]).save(f"{folder}/{str(epoch)}/{file_name}.png")
