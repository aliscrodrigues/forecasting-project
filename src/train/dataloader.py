import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader:
    """Build a DataLoader from NumPy feature and target arrays."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if len(X) != len(y):
        raise ValueError(
            f"X and y contain different sample counts: {len(X)} != {len(y)}"
        )

    X_tensor = torch.from_numpy(np.array(X, dtype=np.float32, order="C", copy=True))
    y_array = np.array(y, dtype=np.float32, order="C", copy=True)
    if y_array.ndim == 1:
        y_array = y_array.reshape(-1, 1)
    y_tensor = torch.from_numpy(y_array)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def batch_to_device(
    batch: tuple[torch.Tensor, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move one loader batch to the same device used by a PyTorch model."""
    X_batch, y_batch = batch
    non_blocking = device.type == "cuda"
    return (
        X_batch.to(device, non_blocking=non_blocking),
        y_batch.to(device, non_blocking=non_blocking),
    )
