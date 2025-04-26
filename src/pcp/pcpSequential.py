import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.DatasetHelper import PCPDataset
from src.LoopHelper import train_and_test_loops
from src.ModelHelper import SequentialModelPCP

if __name__ == '__main__':
    batch_size = 32
    input_features = 12
    output_classes = 5
    epochs = 50

    dataset = PCPDataset("../../data/pcp/pcpData.csv")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = SequentialModelPCP(input_features, output_classes)
    print(f"Model: {model}\n")

    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_model = train_and_test_loops(train_dataloader, test_dataloader, model, loss_fn, optimizer, batch_size, epochs)
    torch.save(best_model.state_dict(), './pcpSequentialModel.pth')
