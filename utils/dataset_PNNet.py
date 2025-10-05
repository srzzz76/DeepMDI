import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import numpy as np


class ISBI_Loader(Dataset):
    """
    Dataset loader for pre-trained normalization network.
    Loads two input frames and their corresponding normalized labels.
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path1 = glob.glob(os.path.join(data_path, 'frame1/*.png'))
        self.imgs_path2 = glob.glob(os.path.join(data_path, 'frame2/*.png'))

    def augment(self, image, flipCode):
        """
        Simple image augmentation using flipping.
        flipCode: 1 - horizontal, 0 - vertical, -1 - both
        """
        return cv2.flip(image, flipCode)

    def __getitem__(self, index):
        """
        Load and preprocess input images and labels.
        """
        # Load input images
        image_path1 = self.imgs_path1[index]
        image_path2 = self.imgs_path2[index]

        # Corresponding label paths (normalized images)
        label_path1 = image_path1.replace('frame1', 'frame1_n')
        label_path2 = image_path1.replace('frame1', 'frame2_n')

        # Read images
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)
        label1 = cv2.imread(label_path1)
        label2 = cv2.imread(label_path2)

        if label2 is None:
            raise ValueError(f"Failed to load label image at {label_path2}")

        # Convert to grayscale
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        label1 = cv2.cvtColor(label1, cv2.COLOR_BGR2GRAY)
        label2 = cv2.cvtColor(label2, cv2.COLOR_BGR2GRAY)

        # Add channel dimension
        image1 = image1.reshape(1, image1.shape[0], image1.shape[1])
        image2 = image2.reshape(1, image2.shape[0], image2.shape[1])
        label1 = label1.reshape(1, label1.shape[0], label1.shape[1])
        label2 = label2.reshape(1, label2.shape[0], label2.shape[1])

        # Normalize to [0,1]
        image1 = image1 / 65535.0 if image1.max() > 255 else image1 / 255.0
        image2 = image2 / 65535.0 if image2.max() > 255 else image2 / 255.0
        label1 = label1 / 65535.0 if label1.max() > 255 else label1 / 255.0
        label2 = label2 / 65535.0 if label2.max() > 255 else label2 / 255.0

        # Concatenate two frames along channel dimension
        image = np.concatenate((image1, image2), axis=0)
        label = np.concatenate((label1, label2), axis=0)

        return image, label

    def __len__(self):
        """
        Return total number of samples.
        """
        return len(self.imgs_path1)


if __name__ == "__main__":
    dataset = ISBI_Loader("../data/simu_1_2/")
    print("Number of samples:", len(dataset))

    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=15, shuffle=True)

    for image, label in train_loader:
        print("Input batch shape:", image.shape)
        print("Label batch shape:", label.shape)
        break
