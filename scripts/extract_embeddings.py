# general imports
import torch
import hydra
import os
import ollama
import pandas as pd

# text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter

# utils
from pathlib import Path

# typing
from omegaconf import DictConfig
from os import PathLike
from typing import List


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="extract_embeddings"
)
def main(cfg: DictConfig) -> None:

    ROOT: PathLike = Path(cfg.paths.root)
    input_path: PathLike = ROOT / cfg.paths.input
    output_path: PathLike = ROOT / cfg.paths.output

    df_descr: pd.DataFrame = pd.read_csv(input_path)
    df: pd.DataFrame = pd.DataFrame(
        columns=["image", "chunk_id", "chunk", "embedding_file"])

    for i in df_descr.index:
        image: str = df_descr.loc[i, "image"]
        descr: str = df_descr.loc[i, "description"]

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"],
            chunk_size=256,
            chunk_overlap=0,
            length_function=len,
        )
        chunks: List[str] = text_splitter.split_text(descr)

        for j in range(len(chunks)):
            chunk = chunks[j]
            chunk_embedding: ollama.EmbeddingsResponse = ollama.embeddings(
                model=cfg.embeddings.model,
                prompt=chunk
            )

            chunk_embedding: torch.Tensor = torch.Tensor(
                chunk_embedding.embedding
            )

            embedding_path = ROOT / cfg.paths.embeddings
            os.makedirs(embedding_path, exist_ok=True)

            torch.save(chunk, embedding_path /
                       f"{"_".join(image.split("/")[-1].split("."))}_{j}.pt")
            df.loc[len(df)] = [
                image,
                j,
                chunk,
                str(embedding_path /
                    f"{"_".join(image.split("/")[-1].split("."))}_{j}.pt")
            ]

        df.to_csv(output_path)


if __name__ == "__main__":
    main()
