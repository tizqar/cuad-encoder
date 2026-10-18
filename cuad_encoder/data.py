"""Download CUAD and derive the contract-level splits and the validation file.

The test split is CUAD's own (102 contracts). A tenth of CUAD's 408 training contracts
is held out as validation by a stable hash of the contract title.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

CUAD_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
CUAD_SHA256 = "f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a"
CUAD_FILES = ("train_separate_questions.json", "test.json")

VAL_FRACTION = 0.1
SPLIT_SEED = 3407


def download_cuad(raw_dir: Path) -> None:
    """Fetch CUAD's release archive, check its hash and extract the two SQuAD files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(CUAD_URL) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != CUAD_SHA256:
        raise RuntimeError(f"unexpected CUAD archive: sha256 {digest}")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in CUAD_FILES:
            (raw_dir / name).write_bytes(archive.read(name))


def _titles(squad_json: Path) -> set[str]:
    return {article["title"] for article in json.loads(squad_json.read_text())["data"]}


def val_bucket(title: str, seed: int = SPLIT_SEED) -> float:
    """Stable value in [0, 1] for a contract title."""
    digest = hashlib.sha256(f"{seed}:{title}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def make_splits(
    train_titles: set[str],
    test_titles: set[str],
    val_fraction: float = VAL_FRACTION,
    seed: int = SPLIT_SEED,
) -> dict[str, str]:
    """Assign every contract to train, val or test."""
    splits = {}
    for title in sorted(train_titles | test_titles):
        if title in test_titles:
            splits[title] = "test"
        elif val_bucket(title, seed) < val_fraction:
            splits[title] = "val"
        else:
            splits[title] = "train"
    return splits


def build_val_json(train_json: Path, splits: dict[str, str]) -> dict:
    """The validation contracts in the shape of CUAD's test.json, one entry per question.

    CUAD's training file repeats a question once per gold span; the test file lists all
    spans of a question together, which is what the evaluator expects.
    """
    articles = []
    for article in json.loads(train_json.read_text())["data"]:
        if splits.get(article["title"]) != "val":
            continue
        paragraph = article["paragraphs"][0]
        by_question: dict[str, dict] = {}
        for qa in paragraph["qas"]:
            entry = by_question.setdefault(
                qa["question"],
                {
                    "question": qa["question"],
                    "id": qa["id"].rsplit("__", 1)[0] if "__" in qa["id"] else qa["id"],
                    "answers": [],
                    "is_impossible": True,
                },
            )
            if not qa.get("is_impossible") and qa.get("answers"):
                entry["answers"].extend(qa["answers"])
                entry["is_impossible"] = False
        articles.append(
            {
                "title": article["title"],
                "paragraphs": [
                    {"context": paragraph["context"], "qas": list(by_question.values())}
                ],
            }
        )
    return {"version": "cuad_v1_val", "data": articles}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CUAD and write the splits.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    args = parser.parse_args()

    if not all((args.raw_dir / name).exists() for name in CUAD_FILES):
        print(f"downloading {CUAD_URL}")
        download_cuad(args.raw_dir)

    train_json, test_json = (args.raw_dir / name for name in CUAD_FILES)
    splits = make_splits(_titles(train_json), _titles(test_json))
    args.splits.parent.mkdir(parents=True, exist_ok=True)
    args.splits.write_text(json.dumps(splits, ensure_ascii=False, indent=2))

    val = build_val_json(train_json, splits)
    (args.raw_dir / "val.json").write_text(json.dumps(val, ensure_ascii=False))

    counts = Counter(splits.values())
    print(f"contracts: train {counts['train']}, val {counts['val']}, test {counts['test']}")
    print(f"wrote {args.splits} and {args.raw_dir / 'val.json'}")


if __name__ == "__main__":
    main()
