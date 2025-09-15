import torch

class Silu:
    @classmethod
    def silu(cls, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
