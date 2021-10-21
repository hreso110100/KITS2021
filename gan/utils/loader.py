import os
import random

import numpy as np
import yaml
from PIL import Image


class Loader:

    def __init__(self):
        with open(f"../config/model_config.yaml", 'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        self.dataset_folder = self.config["folders"]["dataset"]
        self.generated_folder = self.config["folders"]["generated"]

        if not os.path.exists(self.generated_folder):
            os.makedirs(self.generated_folder)

        self.case_list = os.listdir(self.dataset_folder)

    def load_batch(self, batch_size=1):
        """
        Load data from given folder.
        :return: Tuple containing original data and corrupted data.
        """

        # Selecting random cases
        chosen_cases = np.random.choice(len(self.case_list), batch_size, replace=False)
        for _, chosen_index in enumerate(chosen_cases):
            batch = []
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

            batch.append(concatenated_img)
            batch.append()
            yield np.array(batch), np.array(noise_mask)

if __name__ == '__main__':
    l = Loader()
    l.load_batch(1)
