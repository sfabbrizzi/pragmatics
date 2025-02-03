# general imports
import torch
import random
import numpy as np

# torch
from torch.nn import Softmax

# typing
from typing import Optional, List

# our
from src.geometry import compute_distances


class Random_Walk():
    def __init__(
        self,
        space: torch.Tensor,
        v0: Optional[int]
    ) -> None:
        """Implements a random walk on a set of points
        in a embedding space.

        Parameters
        ----------
        space : torch.Tensor
            the embeddings of the points.
        v0 : Optional[int]
            the index of the starting point,
            if None the index is chosen randomly.

        Raises
        ------
        ValueError
            v0 must be an integer in the range
            [0, space.shape[0]) or None
        ValueError
            space must have only unique rows,
            apply torch.unique(space, dim=0)
        """

        if space.shape[0] != torch.unique(space, dim=0).shape[0]:
            raise ValueError("space must have only unique rows, "
                             "apply torch.unique(space, dim=0)")
        else:
            self.space = space

        if v0 is None:
            self.v0 = random.randint(0, space.shape[0])
        elif isinstance(v0, int) and (v0 >= 0) and (v0 < space.shape[0]):
            self.v0 = v0
        else:
            raise ValueError(
                "v0 must be an integer in the range "
                "[0, space.shape[0]) or None"
            )
        self.walk_list = [self.v0]

        self.available_index = np.delete(np.arange(space.shape[0]), v0)
        self.softmax = Softmax(dim=0)

    def step(self, v: int) -> int:
        distances = compute_distances(
            self.space[v],
            self.space[self.available_index]
        )
        probabilities = self.softmax(distances)

        v_new = np.random.choice(
            self.available_index,
            p=probabilities.numpy()
        ).item()

        return v_new

    def walk(self, steps: int) -> List[int]:
        if steps > len(self.available_index):
            raise ValueError("too many steps, space not big enough")

        for _ in range(0, steps):
            v_new = self.step(self.walk_list[-1])
            self.walk_list.append(v_new)

            self.available_index = np.delete(
                self.available_index,
                np.where(self.available_index == v_new)[0],
            )

        return self.walk_list


if __name__ == "__main__":

    space: torch.Tensor = torch.Tensor([
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1],
        [65, 0, 99],
        [4, 6, 7],
        [9, 4, 5]
    ])
    space = torch.unique(space, dim=0)

    assert space.shape == (5, 3)

    v0: int = 2

    rw = Random_Walk(space, v0)
    print(rw.walk(2))
    print(rw.walk(2))
