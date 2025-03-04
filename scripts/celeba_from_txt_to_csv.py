# general imports
import pandas as pd
import argparse
import ast

# utils
from pathlib import Path

# typing
from os import PathLike
from argparse import Namespace
from typing import List

# Initialize parser
parser = argparse.ArgumentParser("CelebA from .txt to .csv")

parser.add_argument("--root", default="../data/celeba")
parser.add_argument("--out_path", default="../data/celeba")
parser.add_argument("--txt", default="list_attr_celeba.txt")
parser.add_argument("--csv", default="list_attr_celeba.csv")


def main():
    args: Namespace = parser.parse_args()

    ROOT: PathLike = Path(args.root)
    TXT_PATH: PathLike = ROOT / args.txt
    OUT_PATH: PathLike = Path(args.out_path)
    CSV_PATH: PathLike = OUT_PATH / args.csv

    with open(TXT_PATH, "r") as f_txt:
        no_elements: str = f_txt.readline().strip()
        assert no_elements.isnumeric()
        no_elements: int = ast.literal_eval(no_elements)

        with open(CSV_PATH, "w") as f_csv:
            header: List[str] = ["image_id"] + f_txt.readline().strip().split()
            f_csv.write(",".join(header) + "\n")
            line: str = f_txt.readline().strip()

            while line != "":
                f_csv.write(",".join(line.split()) + "\n")
                line: str = f_txt.readline().strip()

    df = pd.read_csv(CSV_PATH)
    assert len(df) == no_elements
    assert (df.columns == header).all()


if __name__ == "__main__":
    main()
