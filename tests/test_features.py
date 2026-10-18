import random

import pytest

from cuad_encoder.features import FeatureBuilder, balance, label_window, verify

CONTEXT = " ".join(f"word{i}" for i in range(200)) + " The licence is exclusive. " * 2


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-150m")


def examples(mode_answers: list[str | None]) -> list[dict]:
    out = []
    start = 0
    for text in mode_answers:
        found = CONTEXT.index(text, start) if text else -1
        start = found + 1 if text else start
        out.append(
            {
                "contract_id": "c",
                "split": "train",
                "question": 'related to "Exclusivity"',
                "clause_type": "Exclusivity",
                "answer_text": text,
                "answer_start": found,
            }
        )
    return out


def test_label_window_finds_the_covering_tokens():
    offsets = [(0, 0), (0, 3), (4, 8), (9, 12)]
    assert label_window(offsets, 1, 3, 4, 12) == (2, 3)
    assert label_window(offsets, 1, 3, 4, 20) is None


def test_every_positive_reconstructs_its_gold_text(tokenizer):
    builder = FeatureBuilder(tokenizer, max_seq_len=64, doc_stride=16, answer_mode="multi")
    features = builder(
        "c", CONTEXT, examples(["The licence is exclusive.", "The licence is exclusive."])
    )
    positives = [f for f in features if f["is_positive"]]
    assert positives
    assert verify(features)["containment_pass_rate"] == 1.0
    assert all(
        f["start_positions"] == f["end_positions"] == 0 for f in features if not f["is_positive"]
    )


def test_answer_modes_label_a_window_with_two_spans(tokenizer):
    group = examples(["The licence is exclusive.", "The licence is exclusive."])

    def positives(mode: str) -> int:
        builder = FeatureBuilder(tokenizer, max_seq_len=512, doc_stride=128, answer_mode=mode)
        return sum(f["is_positive"] for f in builder("c", CONTEXT, group))

    assert positives("multi") == 2
    assert positives("merged") == 1
    assert positives("separate") == 2


def test_balance_keeps_every_positive():
    features = [{"is_positive": i < 3} for i in range(20)]
    kept = balance(features, neg_ratio=2.0, rng=random.Random(0))
    assert sum(f["is_positive"] for f in kept) == 3
    assert len(kept) == 9
