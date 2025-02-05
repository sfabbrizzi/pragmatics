# general imports
import torch
import ollama
import pandas as pd

# pytorch
from torch.nn import functional as F

# utils
from pathlib import Path

# typing
from os import PathLike
from typing import List, Optional


def load_embedding(
    path: PathLike
) -> torch.Tensor:
    embedding: torch.Tensor = torch.load(path)

    return embedding


def load_embeddings(
    root: PathLike,
    df: pd.DataFrame
) -> torch.Tensor:

    paths: List[PathLike] = [Path(root) / embeddings_file
                             for embeddings_file in df.embedding_file]
    tensors: List[torch.Tensor] = [torch.load(path) for path in paths]

    embedding_space: torch.Tensor = torch.stack(tensors)

    return embedding_space


def query(
    model: str,
    prompt: str,
    embedding_space: torch.Tensor,
    top: Optional[int] = None
) -> torch.Tensor:
    # extract embedding from prompt
    query_embedding: List[float] = ollama.EmbeddingsResponse(
        model=model,
        prompt=prompt
    ).embedding
    query_embedding: torch.Tensor = torch.Tensor(query_embedding)

    # compute similarity
    similarities = F.cosine_similarity(query_embedding, embedding_space)
    _, sorted_indices = torch.sort(similarities, descending=True)

    return sorted_indices[:top]


if __name__ == "__main__":
    space = torch.Tensor([
        [0, 0, 0],
        [1, 1, 1],
        [65, 0, 99],
        [3, 4, 5]
    ])

    assert space.shape == (4, 3)

    v = 3*torch.ones(3)

    sim = F.cosine_similarity(v, space)
    val, ind = torch.sort(sim, descending=True)
    print(space[ind][:3])
