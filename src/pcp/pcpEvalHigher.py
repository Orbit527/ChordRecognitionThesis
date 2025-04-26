import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.DatasetHelper import PCPDataset
from src.LoopHelper import test_loop
from src.ModelHelper import SequentialModelPCP

if __name__ == '__main__':
    batch_size = 50
    input_features = 12
    output_classes = 5

    dataset = PCPDataset("../../data/pcpHigher/pcpData.csv")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    train_dataset, test_dataset = random_split(dataset, [0, len(dataset)])
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SequentialModelPCP(input_features=input_features, output_classes=output_classes)
    model.load_state_dict(torch.load("pcpSequentialModel.pth"))
    print(f"Model: {model}\n")

    loss_fn = nn.CrossEntropyLoss()
    correct_score, cm = test_loop(test_dataloader, model, loss_fn)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(5), yticklabels=range(5))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
