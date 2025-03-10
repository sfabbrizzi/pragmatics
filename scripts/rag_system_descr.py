# general imports
import torch
import ollama
# import os
import pandas as pd
import pydantic
import re
import argparse
import numpy as np

# vqa
# from PIL import Image
# from transformers import pipeline

# utils
from collections import defaultdict
from lightning import seed_everything
from pathlib import Path

# typing
from typing import List
from argparse import Namespace
from os import PathLike
from numpy.typing import ArrayLike

# ours
from src.random_walks import RandomWalk


class Response(pydantic.BaseModel):
    questions: List[str]


# initialize parser
parser = argparse.ArgumentParser("RAG.")

# paths
parser.add_argument("--root", default="../")
parser.add_argument(
    "--images_path",
    default="data/sdxl-turbo/postal_worker"
)
parser.add_argument(
    "--embeddings_path",
    default="data/embeddings/image/sdxl-turbo/postal_worker"
)
parser.add_argument(
    "--df_path",
    default="results/1320_postal_worker_llava-phi3.tsv"
)

parser.add_argument("--seed", default=1917)
parser.add_argument("--n_steps", default=4)
parser.add_argument("--n_iter", default=10)


def main():
    args: Namespace = parser.parse_args()

    # seeding
    seed_everything(args.seed)
    torch.mps.manual_seed(args.seed)

    # Set up
    ROOT: PathLike = Path(args.root)
    df: pd.DataFrame = pd.read_csv(ROOT / args.df_path, sep="|")

    image_emb_space: ArrayLike = np.stack(
        [np.load(ROOT / args.embeddings_path / f"{file_name[:-4]}.npy")
         for file_name in df.image]
    )

    rw: RandomWalk = RandomWalk(space=image_emb_space, v0=None)
    first_walk: bool = True

    report: str = "# Report\n\n"

    for i in range(args.n_iter):
        report += f"## Step {i}\n\n"
        # STEP 1: Random Walk
        if first_walk:
            steps = rw.walk(args.n_steps)
            first_walk = False
        else:
            steps: List[int] = list()
            steps.append(rw.uniform_step())
            steps += rw.walk(
                args.n_steps,
            )[1:]

        print(steps)
        report += "### Walked images:\n"
        for s in steps:
            report += (
                f"![{s}]"
                f"({str(ROOT / args.images_path / df.loc[s, "image"])})"
            )
        report += "\n\n"

        # STEP 2: LLM -> Common Patterns
        cumulative_descr = "\n\n----------\n\n".join(
            df.loc[steps, "description"])

        response: ollama.ChatResponse = ollama.chat(
            model="deepseek-r1:8b",
            messages=[
                {
                    'role': 'system',
                    'content': """Given a series of image descriptions
                        separated by '\\n\\n----------\\n\\n',
                        your task is to formulate hypotheses on the
                        content ofthe dataset these images are from
                        in the form of questions.

                        Example 1 - the images show
                                    people playing with dogs.
                            Questions:
                                1) Does the image show a dog?
                                2) Does the image show a person?
                                3) Are there a dog and a person playing?

                        Example 2 - The images show a person
                                    wearing white shirt and sunglasses.
                            Questions:
                                1) Does person wear jeans?
                                2) Does the person wear sunglasses?
                                3) Is the shirt red?

                        IMPORTANT:
                            - Keep the questions as simple as possible.
                            - avoid using and clauses.
                            - The only allowed punctuation mark is '?'.
                            - MAX 10 words per question.
                        """,
                },
                {
                    'role': 'user',
                    'content': ("Formulate questions from:"
                                f"{cumulative_descr}"),
                }
            ],
            options={"args.seed": args.seed,
                     "num_ctx": 30_000},
            format=Response.model_json_schema()
        )
        response = Response.model_validate_json(response['message']['content'])

        # STEP 4: Formulate Questions
        # STEP 4.2: VQA
        report += "### Answers\n"
        answers: dict = {
            re.sub("[|*[{()}]:]|([0-9][.)])", "", q): defaultdict(int)
            for q in response.questions
        }

        # vqa_pipeline = pipeline(
        #     "visual-question-answering",
        #     model="dandelin/vilt-b32-finetuned-vqa",
        #     device="mps" if torch.backends.mps.is_available() else "cpu",
        #     use_fast=True
        # )

        # for image_name in os.listdir(args.images_path):
        #     if image_name[0] == ".":
        #         continue
        #     if not os.path.isfile(args.images_path / image_name):
        #         continue
        #
        #     image = Image.open(args.images_path / image_name)

        #     vqa_resp = vqa_pipeline(image, q, top_k=1)[0]
        #     if vqa_resp["score"] >= .75:
        #         answers[q][vqa_resp["answer"]] += 1

        for q in answers.keys():
            report += q + " --> "
            # for answ, no in answers[q].items():
            #     report += answ + f": {no}" + "\t"
            report += "\n"

    with open(
        f"../results/walk_report_descr_args.seed_{args.seed}.md", "w"
    ) as f:
        f.write(report)


if __name__ == "__main__":
    main()
