import os

import numpy as np
import yaml
from PIL import Image


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
        # np.random.seed(420)
        chosen_cases = np.random.choice(len(self.case_list), batch_size, replace=False)
        for _, chosen_index in enumerate(chosen_cases):
            batch_img = []
            batch_mask = []

            # load case folder and select one slice
            case_folder = self.case_list[chosen_index]
            img_list = os.listdir(f"{self.dataset_folder}\\{case_folder}\\imaging")
            # np.random.seed(420)
            img_list_index = np.random.choice(len(img_list), 1, replace=False)[0]

            try:
                loaded_img, loaded_mask = self.load_files(case_folder, img_list[img_list_index])
            except FileNotFoundError:
                print(f"LOGGER: Cannot load case {self.case_list[chosen_index]}.")
                continue

            # min max scaling of imaging
            loaded_img = loaded_img / 255.0
            loaded_mask = np.expand_dims(loaded_mask, axis=0)

            batch_img.append(loaded_img.reshape(self.input_shape))
            batch_mask.append(loaded_mask)

            yield np.array(batch_img), np.array(batch_mask), (case_folder + "_" + img_list[img_list_index])

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

    def save_data(self, prediction: np.array, prediction_case_name: str):
        """
        Saving data.
        :param prediction_case_name: Name of the case and imaging.
        :param prediction: Prediction data.
        """
        pred_3 = prediction[3]
        prediction = np.maximum(prediction[1], [prediction[2]])
        prediction = np.maximum(prediction, pred_3)
        # prediction = np.expand_dims(prediction[0], axis=-1)
        Image.fromarray(prediction[0].astype(np.uint8)).save(f"{self.predictions_folder}/{prediction_case_name}")
