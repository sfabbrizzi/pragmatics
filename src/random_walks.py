# general imports
import random
import numpy as np

# scipy and sklearn
from scipy.special import softmax
from sklearn.neighbors import NearestNeighbors

# typing
from typing import Optional, List
from numpy.typing import ArrayLike


class RandomWalk():
    def __init__(
        self,
        space: ArrayLike,
        v0: Optional[int] = None,
        n_neighbours: int = 20
    ) -> None:
        """Implements a random walk on a nearest neighbours set
        of the given starting point in a embedding space.

        Parameters
        ----------
        space : ArrayLike
            the embeddings of the points.
        v0 : Optional[int]
            the index of the starting point,
            if None the index is chosen randomly.
        n_neighbours: int
            the number of nearest neighbours to
            take into account

        Raises
        ------
        ValueError
            v0 must be an integer in the range
            [0, space.shape[0]) or None
        ValueError
            space must have only unique rows,
            apply np.unique(space, axis=0)
        """

        if space.shape[0] != np.unique(space, axis=0).shape[0]:
            raise ValueError("space must have only unique rows, "
                             "apply np.unique(space, axis=0)")
        self.space = space

        if v0 is None:
            self.walked = [random.randint(0, len(space)-1)]
        elif v0 < 0 or v0 >= len(space):
            raise ValueError(
                "v0 must be an integer in the range"
                "[0, space.shape[0]) or None"
            )
        else:
            self.walked = [v0]

        self.nn: NearestNeighbors = NearestNeighbors(
            n_neighbors=min(n_neighbours + 1, len(space))
        ).fit(space)

        _, indices = self.nn.kneighbors(space[self.walked[0]].reshape(1, -1))

        self.available_index: ArrayLike = np.delete(
            indices.reshape(-1),
            0
        )

    def recompute_neighbours(self, v0: int) -> None:
        """Recomputes the Nearest Neighbours

        Parameters
        ----------
        v0 : int
            index of the point to compute the neighbours of
        """
        _, indices = self.nn.kneighbors(self.space[v0].reshape(1, -1))

        self.available_index: ArrayLike = np.delete(
            indices.reshape(-1),
            0
        )

        self.walked.append(v0)

    def uniform_step(self) -> int:
        """performs a uniformm step

        Returns
        -------
        int
            the newly walked index.
        """
        unwalked = np.array(
            [i for i in np.arange(len(self.space)) if i not in self.walked]
        )
        probabilities = np.ones_like(unwalked) / len(unwalked)
        v_new = np.random.choice(
            unwalked,
            p=probabilities
        ).item()
        self.recompute_neighbours(v_new)

        return v_new

    def step(self) -> int:
        """performs a step if the random walk.

        Returns
        -------
        int
            the newly walked index.
        """
        distances = np.linalg.norm(
            self.space[self.available_index] - self.space[self.walked[-1]],
            axis=1
        )

        probabilities = softmax(-distances).reshape(-1)

        v_new = np.random.choice(
            self.available_index,
            p=probabilities
        ).item()

        self.walked.append(v_new)
        self.available_index = np.delete(
            self.available_index,
            np.where(self.available_index == v_new)[0]
        )

        return v_new

    def walk(
            self,
            steps: int
    ) -> List[int]:
        """Performs the random walk.

        Parameters
        ----------
        steps : int
            number of steps.

        Returns
        -------
        List[int]
            the currently walked indices.

        Raises
        ------
        ValueError
            too many steps, the space is not big enough.
        """
        if steps > len(self.available_index):
            raise ValueError("too many steps, the space is not big enough.")

        current_walk = [self.walked[-1]]
        for _ in range(0, steps):
            v_new = self.step()
            current_walk.append(v_new)

        return current_walk


if __name__ == "__main__":
    space = np.array(
        [[0, 0, 0],
         [1, 1, 1],
         [9, 4, 2],
         [9, 4, 1],
         [100, 98, 21],
         [3, 5, 6],
         [0, 0, 1],
         [0, 100, 100],
         [99, 3, 1]]
    )

    print(len(space))

    rw = RandomWalk(space, 0, 4)
    print(rw.walk(2))
