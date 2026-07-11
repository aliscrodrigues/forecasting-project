"""Utilities for exposing preprocessed NumPy arrays to PyTorch."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TorchDataBundle:
    """Datasets, data loaders and compute device used by the pipeline."""

    train_dataset: TensorDataset
    val_dataset: TensorDataset
    test_dataset: TensorDataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    device: torch.device


def get_device() -> torch.device:
    """Select CUDA, Apple MPS, or CPU, in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def numpy_to_tensors(
    X: np.ndarray, y: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a feature matrix and target vector to float32 tensors."""
    if X.ndim != 2:
        raise ValueError(f"X must have shape (samples, features), got {X.shape}")
    if y.ndim not in (1, 2):
        raise ValueError(f"y must have shape (samples,) or (samples, 1), got {y.shape}")
    if len(X) != len(y):
        raise ValueError(f"X and y contain different sample counts: {len(X)} != {len(y)}")

    # pandas can expose read-only NumPy views. Writable, contiguous buffers
    # prevent undefined behavior if a tensor is modified during training.
    X_array = np.array(X, dtype=np.float32, order="C", copy=True)
    y_array = np.array(y, dtype=np.float32, order="C", copy=True)
    X_tensor = torch.from_numpy(X_array)
    y_tensor = torch.from_numpy(y_array)
    return X_tensor, y_tensor


def create_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    """Create a TensorDataset from NumPy features and targets."""
    return TensorDataset(*numpy_to_tensors(X, y))


def prepare_torch_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
) -> TorchDataBundle:
    """Build datasets and loaders for all temporal splits.

    Tensors remain in CPU memory. During training, use ``batch_to_device`` to
    move only the current batch to the selected accelerator, avoiding loading
    the complete dataset into limited GPU memory.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    device = get_device()
    train_dataset = create_dataset(X_train, y_train)
    val_dataset = create_dataset(X_val, y_val)
    test_dataset = create_dataset(X_test, y_test)
    loader_options = {
        "batch_size": batch_size,
        "pin_memory": device.type == "cuda",
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    print("\n─── PyTorch Data Preparation ───")
    print(f"  Device: {device}")
    for name, dataset, loader in (
        ("Train", train_dataset, train_loader),
        ("Validation", val_dataset, val_loader),
        ("Test", test_dataset, test_loader),
    ):
        X_tensor, y_tensor = dataset.tensors
        print(
            f"  {name:<10} X={tuple(X_tensor.shape)} {X_tensor.dtype} | "
            f"y={tuple(y_tensor.shape)} {y_tensor.dtype} | batches={len(loader)}"
        )

    return TorchDataBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )


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
