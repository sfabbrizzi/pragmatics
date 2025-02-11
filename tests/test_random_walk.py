# general imports
import torch
import pytest

# ours
from src.random_walks import RandomWalk


@pytest.fixture
def space() -> torch.Tensor:
    space_tensor = torch.Tensor(
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

    return space_tensor


@pytest.fixture
def space_repeat_row() -> torch.Tensor:
    space_tensor = torch.Tensor(
        [[0, 0, 0],
         [1, 1, 1],
         [9, 4, 2],
         [9, 4, 2],
         [9, 4, 1],
         [100, 98, 21],
         [3, 5, 6],
         [0, 0, 1],
         [0, 100, 100],
         [99, 3, 1]]
    )

    return space_tensor


def test_init(
        space: torch.Tensor,
        space_repeat_row: torch.Tensor
) -> None:

    rw: RandomWalk = RandomWalk(space, 0)
    assert (rw.space == space).all()

    with pytest.raises(ValueError):
        RandomWalk(space_repeat_row, 0)
    with pytest.raises(ValueError):
        RandomWalk(space, len(space)+1)
    with pytest.raises(ValueError):
        RandomWalk(space, len(space))
    with pytest.raises(ValueError):
        RandomWalk(space, -1)
    with pytest.raises(ValueError):
        RandomWalk(space, "check")

    rw_none: RandomWalk = RandomWalk(space)  # v0 = None
    assert isinstance(rw_none.v0, int)


def test_step(space: torch.Tensor) -> None:
    rw: RandomWalk = RandomWalk(space, 0)

    n_available: int = len(rw.available_index)

    new_v: int = rw.step(4)
    new_v1: int = rw.step(rw.v0)
    new_v2: int = rw.step(2, uniform=True)

    assert len(rw.available_index) == n_available
    assert isinstance(new_v, int)
    assert isinstance(new_v1, int)
    assert isinstance(new_v2, int)

    with pytest.raises(ValueError):
        rw.step(81)
    with pytest.raises(ValueError):
        rw.step(-1)
    with pytest.raises(ValueError):
        rw.step("check")


def test_walk(space: torch.Tensor) -> None:
    rw: RandomWalk = RandomWalk(space, 0)
    assert len(rw.available_index) == len(space)-1

    rw.walk(2)
    assert len(rw.available_index) == len(space)-3
    rw.walk(2, first_step_unifrom=True)
    assert len(rw.available_index) == len(space)-5

    with pytest.raises(ValueError):
        rw.walk(5)
    with pytest.raises(ValueError):
        rw.walk(18)

    rw.walk(1)
    assert len(rw.available_index) == len(space)-6
    rw.walk(3)
    assert len(rw.available_index) == 0

    with pytest.raises(ValueError):
        rw.walk(1)
