import numpy as np
import torch


def predict(model: torch.nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Run inference and return 1-D predictions on CPU as NumPy."""
    model.eval()
    features = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        predictions = model(features)
    return predictions.cpu().numpy().squeeze(-1)
