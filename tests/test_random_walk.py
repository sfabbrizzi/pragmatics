# general imports
import numpy as np
import pytest

# typing
from numpy.typing import ArrayLike

# ours
from src.random_walks import RandomWalk


@pytest.fixture
def space() -> ArrayLike:
    space_arr = np.array(
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

    return space_arr


@pytest.fixture
def space_repeat_row() -> ArrayLike:
    space_arr = np.array(
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

    return space_arr


def test_init(
        space: ArrayLike,
        space_repeat_row: ArrayLike
) -> None:

    rw: RandomWalk = RandomWalk(space, 0)
    assert (rw.space == space).all()
    assert len(rw.available_index) == len(space)-1

    rw: RandomWalk = RandomWalk(space, 0, 4)
    assert len(rw.available_index) == 4

    with pytest.raises(ValueError):
        RandomWalk(space_repeat_row, 0)
    with pytest.raises(ValueError):
        RandomWalk(space, len(space)+1)
    with pytest.raises(ValueError):
        RandomWalk(space, len(space))
    with pytest.raises(ValueError):
        RandomWalk(space, -1)
    with pytest.raises(TypeError):
        RandomWalk(space, "check")

    rw_none: RandomWalk = RandomWalk(space, None)  # v0 = None
    assert isinstance(rw_none.walked[0], int)


def test_step(space: ArrayLike) -> None:
    rw: RandomWalk = RandomWalk(space, 0, 4)

    n_available: int = len(rw.available_index)

    new_v: int = rw.step()
    assert len(rw.available_index) == n_available-1
    new_v1: int = rw.step()
    assert len(rw.available_index) == n_available-2
    new_v2: int = rw.step()
    assert len(rw.available_index) == n_available-3

    assert isinstance(new_v, int)
    assert isinstance(new_v1, int)
    assert isinstance(new_v2, int)


def test_walk(space: ArrayLike) -> None:
    rw: RandomWalk = RandomWalk(space, 0)
    assert len(rw.available_index) == len(space)-1

    rw.walk(2)
    assert len(rw.available_index) == len(space)-3
    rw.walk(2)
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


def test_random_step(space: ArrayLike) -> None:
    rw: RandomWalk = RandomWalk(space, 0, 4)
    rw.step()
    assert len(rw.walked) == 2
    rw.uniform_step()
    assert len(rw.walked) == 3
    assert len(set(rw.walked)) == 3
