# general imports
import torch
import random

# typing
from typing import Optional, List

# our
# from src.geometry import compute_distances


class Random_Walk():
    def __init__(
        self,
        space: torch.Tensor,
        v0: Optional[int]
    ) -> None:
        """_summary_

        Parameters
        ----------
        space : torch.Tensor
            _description_
        v0 : Optional[int]
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        self.space = space

        if v0 is None:
            self.v0 = random.randint(0, space.shape[0])
        elif isinstance(v0, int) and (v0 >= 0) and (v0 < space.shape[0]):
            self.v0 = v0
        else:
            raise ValueError(
                "v0 must be an integer in the range "
                "[0, space.shape[0]] or None"
            )

    def walk(self, steps: int) -> List[int]:
        pass
