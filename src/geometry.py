# general imports
import torch
import os

# utils
from pathlib import Path

# typing
from os import PathLike
from typing import List


def compute_distances(
    v: torch.Tensor,
    space: torch.Tensor
) -> torch.Tensor:

    return torch.linalg.norm(space-v, axis=1)


def load_space(path: PathLike) -> torch.Tensor:
    path: PathLike = Path(path)
    tensors: List[torch.Tensor] = [
        torch.load(path / name)
        for name in os.listdir(path)
        if name[0] != "."
    ]

    space = torch.stack(tensors)
    return space


if __name__ == "__main__":
    space = torch.Tensor([
        [0, 0, 0],
        [1, 1, 1],
        [65, 0, 99]
    ])

    assert space.shape == (3, 3)

    v = torch.ones(3)

    print(compute_distances(v, space))

    space1 = load_space(
        "../data/embeddings/sdxl-turbo/postal_worker"
    )

    print(space1.shape)
    assert space1.shape[0] == 100
