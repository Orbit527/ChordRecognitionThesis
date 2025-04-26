import os

import pandas
import torch
import torchaudio
from PIL import Image
from torch.utils.data import Dataset

from LabelHelper import chordToLabel


class RawDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform

        self.files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file_name = self.files[item]

        label = chordToLabel[file_name.split('-')[0]]

        audio_path = os.path.join(self.folder_path, file_name)
        waveform, sample_rate = torchaudio.load(audio_path)

        duration = 1  # 1 second
        max_samples = duration * sample_rate
        waveform = waveform[:, :max_samples]

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label


class PCPDataset(Dataset):
    def __init__(self, csvFile):
        df = pandas.read_csv(csvFile, header=None)

        self.labels = df.iloc[:, 0]
        self.features = df.iloc[:, 1:]

        # converting to values is done, to get labels and features into a numpy array
        self.labels = torch.tensor(self.labels.values, dtype=torch.long)
        self.features = torch.tensor(self.features.values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.features[item], self.labels[item]


class SpectrogramDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, item):
        image_name = self.image_filenames[item]
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert('RGB')

        label = int(image_name.split('_')[0])

        if self.transform:
            image = self.transform(image)

        return image, label
