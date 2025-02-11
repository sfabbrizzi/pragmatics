# general imports
import torch
import ollama
import os
import pandas as pd

# pytorch
from torch.nn import functional as F

# utils
from pathlib import Path

# typing
from os import PathLike
from typing import List, Optional, Tuple


def load_embedding(
    path: PathLike
) -> torch.Tensor:
    embedding: torch.Tensor = torch.load(path)

    return embedding


def load_embeddings(
    root: PathLike,
    df: Optional[pd.DataFrame] = None
) -> torch.Tensor:
    if df is None:
        files = [f for f in os.listdir(root)
                 if f[-3:] == ".pt" and f[0] != "."]
        print(files[133])
    else:
        if "embedding_file" not in df.columns:
            raise ValueError("df must have a embedding_file columns")
        files = df.embedding_file

    paths: List[PathLike] = [Path(root) / embeddings_file
                             for embeddings_file in files]
    tensors: List[torch.Tensor] = [torch.load(path) for path in paths]

    embedding_space: torch.Tensor = torch.stack(tensors)

    return embedding_space


def query(
    model: str,
    prompt: str,
    embedding_space: torch.Tensor,
    top: Optional[int] = None
) -> Tuple[torch.Tensor]:
    # extract embedding from prompt
    query_embedding: List[float] = ollama.embed(
        model=model,
        input=prompt
    ).embeddings
    query_embedding: torch.Tensor = torch.Tensor(query_embedding)

    # compute similarity
    similarities = F.cosine_similarity(query_embedding, embedding_space)
    sorted_sim, sorted_indices = torch.sort(similarities, descending=True)

    return sorted_sim[:top], sorted_indices[:top],
