import json

from cuad_encoder.clause_types import CLAUSE_TYPES, clause_type_from_question
from cuad_encoder.data import build_val_json, make_splits, val_bucket


def test_val_bucket_is_stable_and_in_range():
    assert val_bucket("contract") == val_bucket("contract")
    assert val_bucket("contract", seed=1) != val_bucket("contract", seed=2)
    assert all(0.0 <= val_bucket(f"c{i}") <= 1.0 for i in range(100))


def test_splits_keep_the_test_contracts_and_hold_out_a_tenth():
    train = {f"train-{i}" for i in range(2000)}
    test = {"test-1", "test-2"}
    splits = make_splits(train, test)
    assert splits["test-1"] == splits["test-2"] == "test"
    n_val = sum(v == "val" for v in splits.values())
    assert 150 < n_val < 250


def test_val_json_merges_the_spans_of_one_question(tmp_path):
    q = 'Highlight the parts (if any) of this contract related to "Parties"'
    qas = [
        {"id": "c__Parties_0", "question": q, "answers": [{"text": "A", "answer_start": 0}]},
        {"id": "c__Parties_1", "question": q, "answers": [{"text": "B", "answer_start": 2}]},
        {"id": "c__Law", "question": "other", "answers": [], "is_impossible": True},
    ]
    train = {"data": [{"title": "c", "paragraphs": [{"context": "A B", "qas": qas}]}]}
    path = tmp_path / "train.json"
    path.write_text(json.dumps(train))

    val = build_val_json(path, {"c": "val"})["data"]
    merged = val[0]["paragraphs"][0]["qas"]
    assert [a["text"] for a in merged[0]["answers"]] == ["A", "B"]
    assert merged[1]["is_impossible"] is True
    assert build_val_json(path, {"c": "train"})["data"] == []


def test_clause_type_is_read_from_the_quoted_name():
    assert len(CLAUSE_TYPES) == 41
    q = 'Highlight the parts (if any) of this contract related to "Affiliate License-Licensee"'
    assert clause_type_from_question(q) == "Affiliate License-Licensee"
    assert clause_type_from_question("What is the governing law?") == "Governing Law"
    assert clause_type_from_question("unrelated") is None
