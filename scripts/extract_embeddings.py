# general imports
import torch
import hydra
import os

# transformers
from transformers import CLIPImageProcessor, CLIPModel

# images
from PIL import Image

# utils
from pathlib import Path

# typing
from omegaconf import DictConfig
from os import PathLike
from numpy.typing import ArrayLike


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="extract_embeddings"
)
def main(cfg: DictConfig) -> None:
    device: str = "mps"
    torch_dtype: torch.dtype = torch.float16

    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        device_map=device,
        torch_dtype=torch_dtype,
    )

    processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-base-patch32")

    ROOT: PathLike = Path(cfg.root)
    input_path: PathLike = ROOT / cfg.input

    output_path: PathLike = ROOT / cfg.output
    os.makedirs(output_path, exist_ok=True)

    for file_name in os.listdir(input_path):
        if file_name[-4:] != ".png" or file_name[0] == ".":
            continue
        image: Image = Image.open(input_path / file_name)

        inputs: ArrayLike = processor(
            image
        )["pixel_values"][0].reshape(1, 3, 224, -1)
        inputs: torch.Tensor = torch.from_numpy(inputs)

        with torch.no_grad():
            outputs: torch.Tensor = model.get_image_features(
                inputs.to(device)
            ).cpu().reshape(-1)

            torch.save(outputs, output_path / f"{file_name[:-4]}.pt")

    print(f"extraction of {cfg.input} done!")


if __name__ == "__main__":
    main()
