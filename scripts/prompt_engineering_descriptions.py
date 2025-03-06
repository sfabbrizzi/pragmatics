# general imports
import ollama
import argparse
import os
import re

# utils
from pathlib import Path
from tqdm import tqdm

# typing
from argparse import Namespace
from os import PathLike


# Initialize parser
parser = argparse.ArgumentParser("Extract descriptions from the images.")

# paths
parser.add_argument("--root", default="../")
parser.add_argument(
    "--input_path",
    default="data/sdxl-turbo/postal_worker")
parser.add_argument(
    "--output_path",
    default="results"
)
# generation
parser.add_argument("--model", default="llava-phi3")
parser.add_argument("--prompt", default="What is in this image?")
parser.add_argument("--seed", default=1320)


def main() -> None:

    args: Namespace = parser.parse_args()

    ROOT: PathLike = Path(args.root)
    input_path: PathLike = ROOT / args.input_path
    output_path: PathLike = ROOT / args.output_path / \
        f"{args.seed}_postal_worker_llava-phi3.tsv"

    with open(output_path, "w") as f:
        header: str = "image|description\n"
        f.write(header)

        prompt = """
            What is in the image?

            Add as many details as possible.

            Describe the people in the image and their activities.
        """

        for image_path in tqdm(os.listdir(input_path)):
            if image_path[0] == ".":
                continue
            if not os.path.isfile(input_path / image_path):
                continue
            response: ollama.ChatResponse = ollama.chat(
                model=args.model,
                messages=[
                    # {
                    #     'role': 'system',
                    #     'content': SYSTEM_PROMPT,
                    # },
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [input_path / image_path]
                    }
                ],
                options={"seed": args.seed}
            )

            description: str = re.sub(
                pattern="[\\n\\t|*]|([0-9][).])",
                repl=" ",
                string=response.message.content
            ).strip()
            description: str = " ".join(
                [w for w in description.split() if w != ""]
            )
            line: str = f"{image_path}|{description}\n"
            f.write(line)


if __name__ == "__main__":

    main()
