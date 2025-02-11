# general imports
import pytest
import pandas as pd

# utils
from pathlib import Path

# typing
from os import PathLike

# ours
from src.queries import (
    load_embedding,
    load_embeddings,
    query
)


@pytest.fixture
def vector_path() -> PathLike:
    path = Path(
        "./data/embeddings/text/sdxl-turbo/postal_worker/4_png_0.pt"
    )

    return path


@pytest.fixture
def space_path() -> PathLike:
    path = Path(
        "./data/embeddings/text/sdxl-turbo/postal_worker"
    )

    return path


@pytest.fixture
def df() -> pd.DataFrame:
    df = pd.DataFrame(columns=["foo", "embedding_file"])
    for i in range(35):
        df.loc[len(df)] = [f"{i}", f"{i}_png_0.pt"]

    return df


def test_load_embedding(
    vector_path: PathLike
) -> None:
    emb = load_embedding(vector_path)
    assert emb.shape == (1024,)


def test_load_embeddings_no_df(
    space_path: PathLike
) -> None:
    emb = load_embeddings(space_path)
    assert emb.shape == (483, 1024)


def test_load_embeddings_df(
    space_path: PathLike,
    df: pd.DataFrame
) -> None:
    emb = load_embeddings(space_path, df=df)
    assert emb.shape == (35, 1024)

    wrong_df = pd.DataFrame(columns=["Foo"])
    wrong_df.Foo = [1, 2, 3, 4, 5]

    with pytest.raises(ValueError):
        load_embeddings(space_path, df=wrong_df)


def test_query(
    space_path: PathLike,
) -> None:
    space = load_embeddings(space_path)
    prompt = "hello world!"
    model = "mxbai-embed-large"

    sim, result = query(model, prompt, space)
    sim_top_10, result_top_10 = query(model, prompt, space, top=10)

    assert result.shape == (483,)
    assert space[result].shape == (483, 1024)
    assert sim.shape == (483,)
    assert result_top_10.shape == (10,)
    assert space[result_top_10].shape == (10, 1024)
    assert sim_top_10.shape == (10,)

    assert (result_top_10 == result[:10]).all()
    assert (sim_top_10 == sim[:10]).all()
