import torch.nn as nn


class NeuralModel(nn.Module):
    """Feed-forward MLP for tabular demand forecasting (Linear → ReLU → Dropout)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128, 64, 32),
        dropout: float = 0.20,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
