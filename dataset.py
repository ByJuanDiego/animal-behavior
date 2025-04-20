import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


num_classes = 140  # Número de clases en el dataset


def get_labels(train_labels_csv):
    train_data = pd.read_csv(train_labels_csv)

    # Inicializar un array para las etiquetas en formato one-hot
    one_hot_labels = np.zeros((train_data.shape[0], num_classes), dtype=int)

    # Llenar las etiquetas one-hot
    for i, predicted in enumerate(train_data['Predicted']):
        predicted_labels = map(int, predicted.split())
        for label in predicted_labels:
            if label < num_classes:
                one_hot_labels[i, label] = 1

    # Crear el DataFrame con las etiquetas one-hot
    one_hot_labels_df = pd.DataFrame(one_hot_labels, columns=[f"Label_{cls}" for cls in range(num_classes)])

    # Concatenar las etiquetas one-hot con el DataFrame original
    train_data = pd.concat([train_data, one_hot_labels_df], axis=1)

    return train_data


class AnimalVideoDataset(Dataset):
    def __init__(self, video_dir, label_data, transform=None):
        self.video_dir = video_dir
        self.label_data = label_data
        self.transform = transform

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        video_id = self.label_data.iloc[idx]['Id']
        labels = self.label_data.iloc[idx, 2:].to_numpy(dtype=np.float32)

        # Cargar los frames procesados desde el archivo .npy
        video_path = os.path.join(self.video_dir, f"{video_id}.npy")
        frames = np.load(video_path)  # Forma: (16, H, W, 1)

        frames = torch.tensor(frames, dtype=torch.float32)  # Convertir los frames a tensor
        label = torch.tensor(labels, dtype=torch.float32)  # Convertir las etiquetas a tensor

        return frames, label

