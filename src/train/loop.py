import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.pytorch import batch_to_device, get_device


def _mean_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    total = 0.0
    n_samples = 0
    for batch in loader:
        X_batch, y_batch = batch_to_device(batch, device)
        loss = criterion(model(X_batch), y_batch)
        total += loss.item() * len(X_batch)
        n_samples += len(X_batch)
    return total / n_samples if n_samples else float("nan")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device | None = None,
    epochs: int = 10,
    learning_rate: float = 0.001,
) -> nn.Module:
    """Train a PyTorch model, moving batches to CUDA, MPS, or CPU as available."""
    if device is None:
        device = get_device()

    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"  Training on device: {device}")

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        n_train = 0
        for batch in train_loader:
            X_batch, y_batch = batch_to_device(batch, device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(X_batch)
            n_train += len(X_batch)
        train_loss = train_loss_sum / n_train if n_train else float("nan")

        model.eval()
        with torch.no_grad():
            val_loss = _mean_loss(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}"
        )

    return model
