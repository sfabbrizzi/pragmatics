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


class RandomWalk():
    def __init__(
        self,
        space: torch.Tensor,
        v0: Optional[int] = None
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
            self.v0 = random.randint(0, space.shape[0]-1)
        elif isinstance(v0, int) and (v0 >= 0) and (v0 < space.shape[0]):
            self.v0 = v0
        else:
            raise ValueError(
                "v0 must be an integer in the range "
                "[0, space.shape[0]) or None"
            )
        self.walk_list = [self.v0]

        self.available_index = np.delete(np.arange(space.shape[0]), self.v0)
        self.softmax = Softmax(dim=0)

    def step(self, v: int, uniform: bool = False) -> int:
        """performs a step strating from a given
        point in space.

        Parameters
        ----------
        v : int
            index of the starting point.

        uniform : bool, default False
            if true the step next step is drawn
            from uniform probability.

        Returns
        -------
        int
            index of the new point reached.

        Raises
        ------
        ValueError
            the index v must be in [0, self.space.shape[0]).
        """
        if v not in range(0, self.space.shape[0]):
            raise ValueError("the index v must be in [0, self.space.shape[0])")

        if not uniform:
            distances = compute_distances(
                self.space[v],
                self.space[self.available_index]
            )
            probabilities = self.softmax(-distances)

            v_new = np.random.choice(
                self.available_index,
                p=probabilities.numpy()
            ).item()
        else:
            v_new = np.random.choice(
                self.available_index,
                p=np.ones_like(self.available_index) /
                len(self.available_index)
            ).item()

        return v_new

    def walk(
            self,
            steps: int,
            first_step_unifrom: bool = False
    ) -> List[int]:
        """perform a walk of a given numebr of steps
        starting from the last registered step.

        Parameters
        ----------
        steps : int
            number of steps.

        first_step_uniform: bool, default False
            if true, first  is drawn uniformely.

        Returns
        -------
        List[int]
            returns the list of steps done

        Raises
        ------
        ValueError
            raises error if too many steps are requested and
            the space is not big enough.
        """
        if steps > len(self.available_index):
            raise ValueError("too many steps, the space is not big enough.")

        current_walk = [self.walk_list[-1]]

        for i in range(0, steps):
            if i == 0:
                v_new = self.step(
                    self.walk_list[-1],
                    uniform=first_step_unifrom
                )
            else:
                v_new = self.step(
                    self.walk_list[-1],
                    uniform=False)
            self.walk_list.append(v_new)
            current_walk.append(v_new)

            self.available_index = np.delete(
                self.available_index,
                np.where(self.available_index == v_new)[0],
            )

        return current_walk
