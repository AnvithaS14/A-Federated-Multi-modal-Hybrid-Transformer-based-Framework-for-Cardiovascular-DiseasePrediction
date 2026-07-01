import os
import cv2
import torch
import numpy as np
import pandas as pd

from models.cnn import CNNFeatureExtractor


def load_ecg_images(base_path, image_size=64):

    dataset = []

    folders = ['normal', 'disease']

    for label, folder in enumerate(folders):

        folder_path = os.path.join(base_path, folder)

        for image_name in os.listdir(folder_path):

            image_path = os.path.join(folder_path, image_name)

            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            image = cv2.resize(
                image,
                (image_size, image_size)
            )

            image = image / 255.0

            dataset.append((image, label))

    return dataset


def convert_to_csv(base_path):

    dataset = load_ecg_images(base_path)

    cnn_model = CNNFeatureExtractor()

    feature_list = []

    for image, label in dataset:

        image_tensor = torch.tensor(image).float()

        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():

            features = cnn_model(image_tensor)

        features = features.numpy().flatten()

        row = np.append(features, label)

        feature_list.append(row)

    dataframe = pd.DataFrame(feature_list)

    dataframe.to_csv(
        "client1_data.csv",
        index=False
    )

    print("Client 1 CSV Generated Successfully")

    return dataframe