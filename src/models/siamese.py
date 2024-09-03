import torch
from pandas import DataFrame
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm


class SiameseNN(nn.Module):
    def __init__(self):
        super(SiameseNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward_one(self, x):
        return self.fc(x)

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2


def train(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
):
    """
    Train a Siamese PyTorch model.

    Args:
    model (nn.Module): The PyTorch model to train
    train_loader (DataLoader): DataLoader for training data
    criterion (nn.Module): Loss function
    optimizer (optim.Optimizer): Optimizer for training
    num_epochs (int): Number of epochs to train
    device (torch.device): Device to train on (CPU or GPU)

    Returns:
    model (nn.Module): Trained   model
    """
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Training loop with progress bar
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True
        )
        for embedding1, embedding2, target in train_pbar:
            embedding1, embedding2, target = (
                embedding1.to(device),
                embedding2.to(device),
                target.to(device),
            )

            optimizer.zero_grad()
            output1, output2 = model(embedding1, embedding2)
            loss = criterion(output1, output2, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * embedding1.size(0)
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}")

    return model
