"""Compare baseline and length-aware extractive QA."""

import json
import math
import re
import string
from collections import Counter
from pathlib import Path

import torch

from rag.qa_model import QAModel


DATASET_PATH = Path("evaluation/qa_dataset.json")
ALPHAS = [0.0, 2.0]
MAX_ANSWER_TOKENS = 30


def normalize_answer(text):
    """Normalize text before evaluation."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    return " ".join(text.split())


def exact_match(prediction, reference):
    return int(
        normalize_answer(prediction)
        == normalize_answer(reference)
    )


def token_f1(prediction, reference):
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()

    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)

    common = Counter(predicted_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)

    return 2 * precision * recall / (precision + recall)


def get_candidates(qa, question, context):
    inputs = qa.tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=512,
    )

    with torch.no_grad():
        outputs = qa.model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    sequence_ids = inputs.sequence_ids(0)

    context_indices = [
        i for i, sid in enumerate(sequence_ids)
        if sid == 1
    ]

    candidates = []

    for start in context_indices:
        for end in context_indices:
            length = end - start + 1

            if length < 1 or length > MAX_ANSWER_TOKENS:
                continue

            base_score = (
                start_logits[start].item()
                + end_logits[end].item()
            )

            answer = qa.tokenizer.decode(
                inputs["input_ids"][0][start:end + 1],
                skip_special_tokens=True,
            ).strip()

            candidates.append({
                "answer": answer,
                "length": length,
                "base_score": base_score,
            })

    return candidates


def select_answer(candidates, alpha):
    if not candidates:
        return ""

    best = max(
        candidates,
        key=lambda candidate: (
            candidate["base_score"]
            + alpha * math.log1p(candidate["length"])
        ),
    )

    return best["answer"]


def main():
    with DATASET_PATH.open(encoding="utf-8") as file:
        dataset = json.load(file)

    print("Loading QA model...")
    qa = QAModel()

    results = {alpha: [] for alpha in ALPHAS}

    for index, example in enumerate(dataset, start=1):
        question = example["question"]
        context = example["context"]
        reference = example["answer"]

        candidates = get_candidates(qa, question, context)

        print(f"\n{'=' * 65}")
        print(f"QUESTION {index}: {question}")
        print(f"REFERENCE: {reference}")

        for alpha in ALPHAS:
            prediction = select_answer(candidates, alpha)

            em = exact_match(prediction, reference)
            f1 = token_f1(prediction, reference)

            results[alpha].append({
                "em": em,
                "f1": f1,
            })

            print(f"\nALPHA = {alpha}")
            print(f"Prediction: {prediction}")
            print(f"Exact Match: {em}")
            print(f"Token F1: {f1:.4f}")

    print(f"\n{'=' * 65}")
    print("FINAL EVALUATION RESULTS")
    print(f"{'=' * 65}")

    for alpha in ALPHAS:
        scores = results[alpha]

        avg_em = sum(item["em"] for item in scores) / len(scores)
        avg_f1 = sum(item["f1"] for item in scores) / len(scores)

        print(f"\nALPHA = {alpha}")
        print(f"Exact Match: {avg_em * 100:.2f}%")
        print(f"Token F1: {avg_f1 * 100:.2f}%")

    print(f"\nTotal questions: {len(dataset)}")


if __name__ == "__main__":
    main()
