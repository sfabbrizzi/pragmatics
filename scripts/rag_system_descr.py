# general imports
import torch
import ollama
import os
import pandas as pd
import pydantic
import re

# vqa
from PIL import Image
from transformers import pipeline

# utils
from pathlib import Path
from collections import defaultdict
from lightning import seed_everything

# typing
from os import PathLike
from typing import List, Literal

# ours
from src.random_walks import RandomWalk


class Response(pydantic.BaseModel):
    questions: List[str]


class Answer(pydantic.BaseModel):
    answer: Literal["yes", "no", "I do not know"]


# PATHS
ROOT: PathLike = Path("../")
# IMAGES: PathLike = ROOT / "data/sdxl-turbo/postal_worker"
# IMAGE_EMB: PathLike = ROOT / "data/embeddings/image/sdxl-turbo"
# DF: PathLike = ROOT / "results/postal_worker_descriptions_llava-phi3.csv"
# DF_CHUNKS: PathLike = ROOT / "results/postal_worker_chunks.csv"

IMAGES: PathLike = ROOT / "data/sdxl-turbo/various_nationalities"
IMAGE_EMB: PathLike = ROOT / "data/embeddings/image/sdxl-turbo"
DF: PathLike = (ROOT /
                "results/various_nationalities_descriptions_llava-phi3.csv")

# OTHERS
SEED: int = 1_917
N_STEPS: int = 4
MAX_ITER: int = 5

# Seeding
seed_everything(SEED)
torch.mps.manual_seed(SEED)

# Set up
df = pd.read_csv(DF)

image_emb_space: torch.Tensor = torch.stack(
    [torch.load(IMAGE_EMB / f"{file_name[:-4]}.pt")
     for file_name in df.image]
)

rw: RandomWalk = RandomWalk(space=image_emb_space, v0=None)
first_walk: bool = True

report: str = "# Report\n\n"

for i in range(MAX_ITER):
    report += f"## Step {i}\n\n"
    # STEP 1: Random Walk
    if first_walk:
        steps = rw.walk(N_STEPS)
        first_walk = False
    else:
        steps = rw.walk(
            N_STEPS + 1,
            first_step_unifrom=True
        )[1:]
    print(steps)
    report += "### Walked images:\n"
    for s in steps:
        report += (f"![{s}]"
                   f"(../data/sdxl-turbo/{df.loc[s, "image"]})")
    report += "\n\n"

    # STEP 2: Retrieve text
    # chunks = df_chunks.merge(
    #     df.loc[steps], how="right", on="image").reset_index()

    # STEP 3: LLM -> Common Patterns
    cumulative_descr = "\n\n----------\n\n".join(
        df.loc[steps, "description"])

    response: ollama.ChatResponse = ollama.chat(
        model="deepseek-r1:8b",
        messages=[
            {
                'role': 'system',
                'content': """Given a series of texts separated
                    by '\\n\\n----------\\n\\n' in input, your task
                    is to find the relevant aspects of them.

                    These texts describe a subset of a dataset.
                    Formulate hypotheses on the content of the
                    dataset in the form of questions.

                    IMPORTANT! The questions must allow only answers like
                    'yes' or 'no'.

                    Example 1 - Relevant aspects: the images show
                                    people playing with dogs.
                            Questions:
                                Does the image show a dog?
                                Does the image show a person?
                                Are there a dog and a person playing?

                    Example 2 - Relevant aspects: The images show a person
                                    wearing white shirt and sunglasses.
                            Questions:
                                Does person wear jeans?
                                Does the person wear sunglasses?
                                Is the shirt red?

                    IMPORTANT! 1) Keep the questions as simple as possible.
                        2) The only allowed punctuation mark is '?'.
                        3) MAX 10 words per question.
                    """,
            },
            {
                'role': 'user',
                'content': ("Formulate questions starting from"
                            "the following descriptions:\n"
                            f"{cumulative_descr}"),
            }
        ],
        options={"seed": SEED,
                 "num_ctx": 30_000},
        format=Response.model_json_schema()
    )
    response = Response.model_validate_json(response['message']['content'])

    # STEP 4: Formulate Questions
    # STEP 4.1: Ask user to formulate questions
    questions = response.questions

    # STEP 4.2: VQA
    report += "### Answers\n"
    answers = {re.sub(r"[\(\[{}\[\)]", "", q): defaultdict(int)
               for q in questions}

    vqa_pipeline = pipeline(
        "visual-question-answering",
        model="dandelin/vilt-b32-finetuned-vqa",
        device="mps" if torch.backends.mps.is_available() else "cpu",
        use_fast=True
    )

    for image_name in os.listdir(IMAGES):
        if image_name[0] == ".":
            continue
        if not os.path.isfile(IMAGES / image_name):
            continue
        for q in answers.keys():
            image = Image.open(IMAGES / image_name)

            vqa_resp = vqa_pipeline(image, q, top_k=1)[0]
            if vqa_resp["score"] >= .75:
                answers[q][vqa_resp["answer"]] += 1

    # for image_name in os.listdir(IMAGES):
    #     for q in questions:
    #         response = ollama.chat(
    #             model='llava-phi3',
    #             messages=[{
    #                 'role': 'user',
    #                 'content': q,
    #                 'images': [IMAGES / image_name]
    #             }],
    #             format=Answer.model_json_schema()
    #         )
    #         answ = Answer.model_validate_json(
    #             response['message']['content']
    #         ).answer

    #         answers[q][answ] += 1

    for q in answers.keys():
        report += q + " --> "
        for answ, no in answers[q].items():
            report += answ + f": {no}" + "\t"
        report += "\n"

    # Step 5: Continue walking
    # continue_walk: str = input("Press y to continue walking ")
    # if continue_walk != "y":
    #     break

with open(f"../results/walk_report_descr_seed_{SEED}.md", "w") as f:
    f.write(report)
