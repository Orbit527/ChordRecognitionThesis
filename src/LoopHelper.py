import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torcheval.metrics.functional import multiclass_f1_score, multiclass_precision, multiclass_recall


def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)

    train_loss = 0

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation of Nodes
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= len(dataloader)
    print("Training Metrics:")
    print(f"Training Loss: {train_loss:.4f}")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    accuracy = 0

    predictions = []
    labels = []

    with torch.no_grad():
        # for info: this loop is done for every batch
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

            predicted_class = pred.argmax(dim=1)
            predictions.append(predicted_class)
            labels.append(y)

    predictions = torch.cat(predictions)
    labels = torch.cat(labels)

    f1_score = multiclass_f1_score(predictions, labels, num_classes=5, average="macro")
    multiclass_precision1 = multiclass_precision(predictions, labels, num_classes=5, average="macro")
    multiclass_recall1 = multiclass_recall(predictions, labels, num_classes=5, average="macro")

    cm = confusion_matrix(labels, predictions)

    # average over batches
    test_loss /= num_batches
    accuracy /= size

    print(f"Test Metrics:")
    print(f"Accuracy: {(100 * accuracy):.4f}%")
    print(f"Precision: {multiclass_precision1:.4f}")
    print(f"Recall: {multiclass_recall1:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"Validation Loss: {test_loss:.4f}\n")

    return accuracy, cm


def train_and_test_loops(train_dataloader, test_dataloader, model, loss_fn, optimizer, batch_size, epochs):
    best_correct_score = 0.0
    best_model = None
    best_cm = None

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
        correct_score, cm = test_loop(test_dataloader, model, loss_fn)

        if (correct_score > best_correct_score):
            best_correct_score = correct_score
            best_cm = cm
            best_model = model

    print(f"Best score: {best_correct_score}")
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(5),
                yticklabels=range(5))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    return best_model
