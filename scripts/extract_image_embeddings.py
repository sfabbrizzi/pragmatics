# general imports
import torch
import hydra
import os

# clip
from transformers import CLIPModel, CLIPImageProcessor

# image processing
from PIL import Image

# utils
from pathlib import Path

# typing
from omegaconf import DictConfig
from os import PathLike


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="extract_image_embeddings"
)
def main(cfg: DictConfig) -> None:
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"

    ROOT: PathLike = Path(cfg.paths.root)
    input_path: PathLike = ROOT / cfg.paths.input

    output_path: PathLike = ROOT / cfg.paths.output
    os.makedirs(output_path, exist_ok=True)

    model: CLIPModel = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(device)
    processor: CLIPImageProcessor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-base-patch32",
        return_tensors="pt"
    )

    for file_name in os.listdir(input_path):
        if file_name[0] == ".":
            continue
        if not os.path.isfile(input_path / file_name):
            continue
        image: Image = Image.open(input_path / file_name)
        with torch.no_grad():
            inputs: torch.Tensor = processor(
                image
            )["pixel_values"][0].reshape(1, 3, 224, -1)
            outputs: torch.Tensor = model.get_image_features(
                torch.from_numpy(inputs).to(device)
            ).cpu().reshape(-1)

        torch.save(outputs, output_path / f"{file_name[:-4]}.pt")


if __name__ == "__main__":
    main()
