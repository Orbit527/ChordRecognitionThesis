import torch
import torch.nn as nn
import torch.nn.functional as F


class SequentialModel(nn.Module):
    def __init__(self, input_features, output_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linearReluStack = nn.Sequential(
            nn.Linear(input_features, 1000),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(100, output_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linearReluStack(x)
        return logits


class SequentialModelPCP(nn.Module):
    def __init__(self, input_features, output_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linearReluStack = nn.Sequential(
            nn.Linear(input_features, 10),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(8, output_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linearReluStack(x)
        return logits


class CNNModel(nn.Module):
    def __init__(self, output_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Downsample by 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 19 * 19, 64)
        self.fc2 = nn.Linear(64, output_classes)

        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [B, 16, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))  # -> [B, 32, 56, 56]
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)
        return x


class CNN1DModel(nn.Module):
    def __init__(self, output_classes):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=9, padding=4)
        self.bn2 = nn.BatchNorm1d(32)

        self.fc1 = nn.Linear(32 * (44100 // (4 * 4)), 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, output_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
