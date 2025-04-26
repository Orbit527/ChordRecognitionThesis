import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from src.DatasetHelper import SpectrogramDataset
from src.LoopHelper import test_loop
from src.ModelHelper import CNNModel

if __name__ == '__main__':
    batch_size = 50
    output_classes = 5

    transform = transforms.Compose([
        transforms.Resize((77, 77)),
        transforms.ToTensor()
    ])

    dataset = SpectrogramDataset("../../data/spectrogramHigher", transform=transform)

    train_dataset, test_dataset = random_split(dataset, [0, len(dataset)])
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = CNNModel(output_classes=output_classes)
    model.load_state_dict(torch.load("spectrogramCNNModel.pth"))
    print(f"Model: {model}\n")

    loss_fn = nn.CrossEntropyLoss()
    correct_score, cm = test_loop(test_dataloader, model, loss_fn)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(5), yticklabels=range(5))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
