import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import numpy as np
import matplotlib.pyplot as plt


class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        """
        Dataset loader for UMNet model-driven training.
        Only loads input frames (no labels needed).
        """
        self.data_path = data_path
        self.imgs_path1 = glob.glob(os.path.join(data_path, 'frame1_n/*.png'))
        self.imgs_path2 = glob.glob(os.path.join(data_path, 'frame2_n/*.png'))

    def augment(self, image, flipCode):
        """
        Apply simple flip augmentation.
        flipCode: 1 horizontal, 0 vertical, -1 both.
        """
        return cv2.flip(image, flipCode)

    def __getitem__(self, index):
        """
        Load and preprocess input images.
        """
        image_path1 = self.imgs_path1[index]
        image_path2 = self.imgs_path2[index]

        # Read images
        image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

        if image1 is None or image2 is None:
            raise ValueError(f"Failed to load images at {image_path1} or {image_path2}")

        # Add channel dimension
        image1 = image1.reshape(1, image1.shape[0], image1.shape[1])
        image2 = image2.reshape(1, image2.shape[0], image2.shape[1])

        # Normalize
        image1 = image1 / 65535.0 if image1.max() > 255 else image1 / 255.0
        image2 = image2 / 65535.0 if image2.max() > 255 else image2 / 255.0

        # Concatenate two frames as input
        image = np.concatenate((image1, image2), axis=0)

        return image

    def __len__(self):
        """
        Return total number of samples.
        """
        return len(self.imgs_path1)


if __name__ == "__main__":
    # Example usage
    dataset = ISBI_Loader("../data/simu_1_2/")
    print("Number of samples:", len(dataset))

    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=15, shuffle=True)

    for image in train_loader:
        print("Batch shape:", image.shape)
