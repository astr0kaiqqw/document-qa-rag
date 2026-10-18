"""Extract answers from text using a fine-tuned QA model."""

from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "final_qa_model"
)


class QAModel:
    """Extract answer spans from document chunks."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        model_path = str(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )

        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_path,
            local_files_only=True,
        )

        self.model.eval()

    def answer(self, question: str, context: str) -> str:
        """Extract an answer from the provided context."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        if not context or not context.strip():
            raise ValueError("Context cannot be empty")

        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=512,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # Identify tokens belonging to the context.
        sequence_ids = inputs.sequence_ids(0)

        context_indices = [
            i for i, sequence_id in enumerate(sequence_ids)
            if sequence_id == 1
        ]

        best_score = float("-inf")
        best_span = None

        # Search for the highest-scoring valid answer span.
        max_answer_tokens = 30

        for start in context_indices:
            for end in context_indices:
                if end < start:
                    continue

                if end - start + 1 > max_answer_tokens:
                    continue

                score = (
                    start_logits[start].item()
                    + end_logits[end].item()
                )

                if score > best_score:
                    best_score = score
                    best_span = (start, end)

        if best_span is None:
            return ""

        start, end = best_span

        answer = self.tokenizer.decode(
            inputs["input_ids"][0][start:end + 1],
            skip_special_tokens=True,
        )

        return answer.strip()