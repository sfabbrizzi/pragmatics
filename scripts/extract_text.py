# general imports
import ollama
import hydra
import os
import pandas as pd

# utils
from pathlib import Path

# typing
from omegaconf import DictConfig
from os import PathLike


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="extract_text"
)
def main(cfg: DictConfig) -> None:
    ROOT: PathLike = Path(cfg.paths.root)
    input_path: PathLike = ROOT / cfg.paths.input
    output_path: PathLike = ROOT / cfg.paths.output

    df: pd.DataFrame = pd.DataFrame(columns=["image", "description"])

    for image_path in os.listdir(input_path):
        if image_path[0] == ".":
            continue
        if not os.path.isfile(input_path / image_path):
            continue
        response: ollama.ChatResponse = ollama.chat(
            model=cfg.extraction.model,
            messages=[
                {
                    'role': 'system',
                    'content': cfg.extraction.system_message,
                },
                {
                    'role': 'user',
                    'content': cfg.extraction.prompt,
                    'images': [input_path / image_path]
                }
            ],
            options={"seed": cfg.extraction.seed}
        )

        df.loc[len(df)] = [
            f"{cfg.paths.input.split("/")[-1]}/{image_path}",
            response['message']['content']
        ]

    df.to_csv(output_path)


if __name__ == "__main__":
    main()
