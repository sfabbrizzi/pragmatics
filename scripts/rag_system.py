# general imports
import torch
import ollama
import os
import pandas as pd
import pydantic

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
from src.queries import (
    load_embeddings,
    query
)
from src.random_walks import RandomWalk


class Response(pydantic.BaseModel):
    questions: List[str]


class Answer(pydantic.BaseModel):
    answer: Literal["yes", "no", "I do not know"]


# PATHS
ROOT: PathLike = Path("../")
IMAGES: PathLike = ROOT / "data/sdxl-turbo/postal_worker"
IMAGE_EMB: PathLike = ROOT / "data/embeddings/image/sdxl-turbo"
DF: PathLike = ROOT / "results/postal_worker_descriptions.csv"
DF_CHUNKS: PathLike = ROOT / "results/postal_worker_chunks.csv"

# OTHERS
SEED: int = 1_789
N_STEPS: int = 4
MAX_ITER: int = 4

# Seeding
seed_everything(SEED)
torch.mps.manual_seed(SEED)

# Set up
df = pd.read_csv(DF)
df_chunks = pd.read_csv(DF_CHUNKS)

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
        report += f"![{s}.png](../data/sdxl-turbo/postal_worker/{s}.png)"
    report += "\n\n"

    # STEP 2: Retrieve text
    chunks = df_chunks.merge(
        df.loc[steps], how="right", on="image").reset_index()

    # STEP 3: LLM -> Common Patterns
    space = load_embeddings("./", chunks)
    best_median = 0
    for chunk in chunks.chunk:
        # SELEZIONA IL CHUNK CON COS SIM MEDIANA PIU ALTA
        simil, indices = query(
            model="mxbai-embed-large",
            prompt=chunk,
            embedding_space=space
        )
        if torch.median(simil) >= best_median:
            best_median = torch.median(simil)
            best_chunk = chunk
            most_similar = indices[simil >= .8]

    cumulative_chunks = "\n\n----------\n\n".join(
        chunks.loc[most_similar].chunk)
    # cumulative_chunks = "\n\n----------\n\n".join(chunks.chunk)

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

                    IMPORTANT! The question must allow only answers like
                    'yes', 'no', or 'I do not know'.

                    Example 1 - Relevant aspects: the images show
                                    people playing with dogs.
                            Questions:
                                1) Does the image show a dog?
                                2) Does the image show a person?
                                3) Are there a dog and a person playing?

                    Example 2 - Relevant aspects: The images show a person
                                    wearing white shirt and sunglasses.
                            Questions:
                                1) Does person wear jeans?
                                2) Does the person wear sunglasses?
                                3) Is the shirt red?

                    IMPORTANT! Keep the questions as simple as possible.
                    """,
            },
            {
                'role': 'user',
                'content': f"{cumulative_chunks}",
            }
        ],
        options={"seed": SEED,
                 "num_ctx": 30_000},
        format=Response.model_json_schema()
    )
    response = Response.model_validate_json(response['message']['content'])
    # report += f"### Common traits\n{response.aspects}\n"

    # STEP 4: Formulate Questions
    # STEP 4.1: Ask user to formulate questions
    questions = response.questions

    # STEP 4.2: VQA
    report += "### Answers\n"
    answers = {q: defaultdict(int) for q in questions}

    vqa_pipeline = pipeline(
        "visual-question-answering",
        model="dandelin/vilt-b32-finetuned-vqa",
        device="mps" if torch.backends.mps.is_available() else "cpu",
        use_fast=True
    )

    for image_name in os.listdir(IMAGES):
        for q in questions:
            image = Image.open(IMAGES / image_name)

            vqa_resp = vqa_pipeline(image, q, top_k=1)[0]
            if vqa_resp["score"] >= .9:
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

    for q in questions:
        report += q + " --> "
        for answ, no in answers[q].items():
            report += answ + f": {no}" + "\t"
        report += "\n"

    # Step 5: Continue walking
    # continue_walk: str = input("Press y to continue walking ")
    # if continue_walk != "y":
    #     break

with open(f"../results/walk_report_seed_{SEED}.md", "w") as f:
    f.write(report)
