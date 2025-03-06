# general imports
import torch
import os
import argparse

# clip
from transformers import CLIPModel, CLIPImageProcessor

# image processing
from PIL import Image

# utils
from pathlib import Path
from lightning import seed_everything
from tqdm import tqdm

# typing
from argparse import Namespace
from os import PathLike


# initialize parser
parser = argparse.ArgumentParser("Extract embeddings from the images.")

# paths
parser.add_argument("--root", default="../")
parser.add_argument(
    "--input_path",
    default="data/sdxl-turbo/postal_worker")
parser.add_argument(
    "--output_path",
    default="data/embeddings/image/sdxl-turbo/postal_worker"
)

parser.add_argument("--seed", default=0)


def main() -> None:
    args: Namespace = parser.parse_args()

    seed_everything(args.seed)

    device: str = "mps" if torch.backends.mps.is_available() else "cpu"

    ROOT: PathLike = Path(args.root)
    input_path: PathLike = ROOT / args.input_path

    output_path: PathLike = ROOT / args.output_path
    os.makedirs(output_path, exist_ok=True)

    model: CLIPModel = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(device)
    processor: CLIPImageProcessor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-base-patch32",
        return_tensors="pt"
    )

    for file_name in tqdm(os.listdir(input_path)):
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
