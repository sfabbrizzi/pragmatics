# general imports
import pandas as pd
import argparse
import os
import shutil

# utils
from pathlib import Path
from lightning import seed_everything

# typing
from os import PathLike
from argparse import Namespace

# Initialize parser
parser = argparse.ArgumentParser("CelebA from .txt to .csv")

parser.add_argument("--root", default="../data/celeba")
parser.add_argument("--csv", default="list_attr_celeba.csv")
parser.add_argument("--split_csv", default="celeba_split.csv")
parser.add_argument("--seed", default=1492)


def main():
    args: Namespace = parser.parse_args()

    ROOT: PathLike = Path(args.root)
    IMAGES: PathLike = ROOT / "img_align_celeba"
    CSV_PATH: PathLike = ROOT / args.csv
    OUT_PATH: PathLike = ROOT / args.split_csv
    FOLDER: PathLike = ROOT / "split"
    os.makedirs(FOLDER, exist_ok=True)

    SEED: int = args.seed
    seed_everything(SEED)

    df: pd.DataFrame = pd.read_csv(CSV_PATH)
    male: pd.DataFrame = df[df["Male"] == 1]
    female: pd.DataFrame = df[df["Male"] == -1]

    split: pd.DataFrame = pd.concat(
        [
            male[(male["Smiling"] == 1) & (
                male["Eyeglasses"] == 1)].sample(170),
            male[(male["Smiling"] == -1) & (
                male["Eyeglasses"] == 1)].sample(30),
            male[(male["Smiling"] == 1) & (
                male["Eyeglasses"] == -1)].sample(30),
            male[(male["Smiling"] == -1) & (
                male["Eyeglasses"] == -1)].sample(20),
            female[(female["Wearing_Hat"] == 1) & (
                female["Wearing_Necklace"] == 1)].sample(170),
            female[(female["Wearing_Hat"] == -1) &
                   (female["Wearing_Necklace"] == 1)].sample(30),
            female[(female["Wearing_Hat"] == 1) & (
                female["Wearing_Necklace"] == -1)].sample(30),
            female[(female["Wearing_Hat"] == -1) &
                   (female["Wearing_Necklace"] == -1)].sample(20),
        ]
    )

    split.reset_index(inplace=True)

    split.to_csv(OUT_PATH, index=False)

    for file_name in os.listdir(IMAGES):
        if file_name[0] == "." or file_name[-4:] != ".jpg":
            continue

        if file_name in list(split.image_id):
            shutil.copyfile(IMAGES / file_name, FOLDER / file_name)


if __name__ == "__main__":
    main()
