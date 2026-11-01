"""Local extractive QA with original-text spans and bounded-memory windows."""
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import unicodedata

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'final_qa_model'


def _word_character(char: str) -> bool:
    return char.isalnum() or char == '_' or unicodedata.category(char).startswith('M')


def valid_boundary(text: str, position: int) -> bool:
    """Reject cuts inside Unicode words and connected identifiers (Zeta-Q7, AB/X)."""
    if position <= 0 or position >= len(text):
        return True
    left, right = text[position - 1], text[position]
    if _word_character(left) and _word_character(right):
        return False
    connectors = "-‐‑/'’"
    if left in connectors and position >= 2:
        if _word_character(text[position - 2]) and _word_character(right):
            return False
    if right in connectors and position + 1 < len(text):
        if _word_character(left) and _word_character(text[position + 1]):
            return False
    return True


@dataclass(frozen=True)
class AnswerCandidate:
    answer: str
    start: int
    end: int
    score: float
    log_score: float
    null_margin: float
    window: int


class QAModel:
    """Scores describe token spans, never probability of factual correctness.

    The SQuAD-v1 checkpoint cannot reliably detect absent answers. An empty list
    only means there was no admissible span, not a calibrated semantic refusal.
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH, *, max_answer_tokens=30,
                 max_length=512, stride=128, whole_words=True):
        if max_answer_tokens <= 0 or max_length < 4 or stride < 0:
            raise ValueError('Invalid QA token budget')
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(str(model_path), local_files_only=True)
        if not self.tokenizer.is_fast:
            raise ValueError('QA requires a fast tokenizer with character offsets')
        if len(self.tokenizer) != self.model.config.vocab_size:
            raise ValueError('Tokenizer vocabulary does not match QA checkpoint')
        self.model.eval()
        self.max_answer_tokens = max_answer_tokens
        self.max_length = min(max_length, self.model.config.max_position_embeddings)
        self.stride = stride
        self.whole_words = whole_words

    def candidates(self, question: str, context: str, *, top_n: int = 5) -> list[dict]:
        """Return unique, traceable spans. Windows are evaluated one at a time."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Question cannot be empty')
        if not isinstance(context, str) or not context.strip():
            raise ValueError('Context cannot be empty')
        if top_n <= 0:
            raise ValueError('top_n must be positive')
        question_length = len(self.tokenizer(question, add_special_tokens=False)['input_ids'])
        capacity = self.max_length - question_length - self.tokenizer.num_special_tokens_to_add(pair=True)
        if capacity <= 0:
            raise ValueError('Question leaves no token budget for context')
        encoded = self.tokenizer(
            question, context, truncation='only_second', max_length=self.max_length,
            stride=min(self.stride, capacity - 1), return_overflowing_tokens=True,
            return_offsets_mapping=True, return_tensors='pt', padding=True,
        )
        offsets = encoded.pop('offset_mapping')
        encoded.pop('overflow_to_sample_mapping', None)
        unique = {}
        device = next(self.model.parameters()).device
        for window in range(len(encoded['input_ids'])):
            inputs = {k: v[window:window + 1].to(device) for k, v in encoded.items()}
            with torch.inference_mode():
                outputs = self.model(**inputs)
            starts = outputs.start_logits[0].detach().cpu()
            ends = outputs.end_logits[0].detach().cpu()
            indices = [i for i, sid in enumerate(encoded.sequence_ids(window))
                       if sid == 1 and inputs['attention_mask'][0, i].item()]
            if not indices:
                continue
            mask = torch.full_like(starts, float('-inf'))
            mask[indices] = 0
            # Conditional token distribution within this window. Not correctness confidence.
            log_starts = torch.log_softmax(starts + mask, dim=0)
            log_ends = torch.log_softmax(ends + mask, dim=0)
            cls = encoded['input_ids'][window].tolist().index(self.tokenizer.cls_token_id)
            null_score = starts[cls].item() + ends[cls].item()
            for start in indices:
                for end in range(start, min(start + self.max_answer_tokens, indices[-1] + 1)):
                    a, b = offsets[window, start, 0].item(), offsets[window, end, 1].item()
                    if not (0 <= a < b <= len(context)) or not context[a:b].strip():
                        continue
                    if self.whole_words and not (valid_boundary(context, a) and valid_boundary(context, b)):
                        continue
                    score = starts[start].item() + ends[end].item()
                    log_score = log_starts[start].item() + log_ends[end].item()
                    if not math.isfinite(log_score):
                        continue
                    candidate = AnswerCandidate(context[a:b], a, b, score, log_score,
                                                score - null_score, window)
                    key = (a, b)
                    if key not in unique or log_score > unique[key].log_score:
                        unique[key] = candidate
        ranked = sorted(unique.values(), key=lambda c: (-c.log_score, c.start, c.end))
        return [asdict(c) for c in ranked[:top_n]]

    def answer(self, question: str, context: str) -> str:
        """Preserve the original string-returning public API."""
        candidates = self.candidates(question, context, top_n=1)
        return candidates[0]['answer'] if candidates else ''
