# general imports
import torch


def compute_distances(
    v: torch.Tensor,
    space: torch.Tensor
) -> torch.Tensor:

    return torch.linalg.norm(space-v, axis=1)


if __name__ == "__main__":
    space = torch.Tensor([
        [0, 0, 0],
        [1, 1, 1],
        [65, 0, 99]
    ])

    assert space.shape == (3, 3)

    v = torch.ones(3)

    print(compute_distances(v, space))
