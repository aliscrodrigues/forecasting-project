import copy
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train.dataloader import batch_to_device
from utils import get_device

logger = logging.getLogger(__name__)

_LOSS_FACTORIES: dict[str, type[nn.Module]] = {
    "mae": nn.L1Loss,
    "mse": nn.MSELoss,
}


def _build_criterion(loss: str) -> nn.Module:
    factory = _LOSS_FACTORIES.get(loss.lower())
    if factory is None:
        supported = ", ".join(sorted(_LOSS_FACTORIES))
        raise ValueError(f"Unsupported loss '{loss}'. Choose one of: {supported}.")
    return factory()


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
    epochs: int = 500,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    patience: int = 50,
    loss: str = "mae",
) -> nn.Module:
    """Train with MAE or MSE loss and restore the best validation checkpoint."""
    if device is None:
        device = get_device()

    model = model.to(device)
    criterion = _build_criterion(loss)
    loss_label = loss.upper()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        logger.info(
            "Epoch %s/%s - train %s: %.4f | val %s: %.4f",
            epoch + 1,
            epochs,
            loss_label,
            train_loss,
            loss_label,
            val_loss,
        )

        if epochs_without_improvement >= patience:
            logger.info("Early stopping at epoch %s", epoch + 1)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
