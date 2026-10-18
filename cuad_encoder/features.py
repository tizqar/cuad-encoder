"""Build windowed, labelled and balanced QA training features from CUAD.

Each (contract, question) is tokenized into overlapping windows, the same way the
evaluator reads a contract. A window is labelled with the gold spans it contains, or with
the [CLS] token when it contains none.

    python -m cuad_encoder.features --tokenizer jhu-clsp/ettin-encoder-150m
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .clause_types import clause_type_from_question
from .windowing import assert_sliding_windows

ANSWER_MODES = ("multi", "merged", "separate")
META_COLUMNS = (
    "contract_id",
    "split",
    "clause_type",
    "window_index",
    "n_windows",
    "is_positive",
    "n_answers_in_window",
)
SAMPLE_SIZE = 40


def load_examples(
    train_json: Path, splits_path: Path, wanted: Iterable[str]
) -> tuple[list[dict], dict[str, str]]:
    """One example per QA entry of the wanted splits, and the contract texts by title."""
    wanted = set(wanted)
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    data = json.loads(Path(train_json).read_text(encoding="utf-8"))["data"]

    examples: list[dict] = []
    contexts: dict[str, str] = {}
    for article in data:
        title = article["title"]
        if title not in splits:
            raise ValueError(f"{title!r} is missing from {splits_path}")
        split = splits[title]
        if split not in wanted:
            continue
        for paragraph in article["paragraphs"]:
            contexts[title] = paragraph["context"]
            for qa in paragraph["qas"]:
                answers = [] if qa.get("is_impossible") else (qa.get("answers") or [])
                examples.append(
                    {
                        "contract_id": title,
                        "split": split,
                        "question": qa["question"],
                        "clause_type": clause_type_from_question(qa["question"]),
                        "answer_text": answers[0]["text"] if answers else None,
                        "answer_start": int(answers[0]["answer_start"]) if answers else -1,
                    }
                )
    return examples, contexts


def _normalise(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", s.lower()).split())


def label_window(
    offsets, ctx_first: int, ctx_last: int, start_char: int, end_char: int
) -> tuple[int, int] | None:
    """Token span of an answer inside a window, or None if the window does not contain it."""
    if not (offsets[ctx_first][0] <= start_char and end_char <= offsets[ctx_last][1]):
        return None
    ts, te = ctx_first, ctx_last
    while ts <= ctx_last and offsets[ts][0] <= start_char:
        ts += 1
    while te >= ctx_first and offsets[te][1] >= end_char:
        te -= 1
    start_tok, end_tok = ts - 1, te + 1
    if start_tok > end_tok or start_tok < ctx_first or end_tok > ctx_last:
        return None
    return start_tok, end_tok


class FeatureBuilder:
    """Turns one contract's examples into labelled window features.

    Answer modes for a window holding several gold spans of one clause type:
    `multi` emits one positive per span; `merged` only the first span; `separate`
    one feature per CUAD QA entry, so the same window can also appear as a negative.
    """

    def __init__(self, tokenizer, max_seq_len: int, doc_stride: int, answer_mode: str):
        if answer_mode not in ANSWER_MODES:
            raise ValueError(f"answer_mode must be one of {ANSWER_MODES}")
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.doc_stride = doc_stride
        self.answer_mode = answer_mode

    def __call__(self, contract_id: str, context: str, examples: list[dict]) -> list[dict]:
        by_question: dict[str, list[dict]] = defaultdict(list)
        for ex in examples:
            by_question[ex["question"]].append(ex)

        features: list[dict] = []
        for question, group in by_question.items():
            group = sorted(group, key=lambda e: e["answer_start"])
            enc = self.tok(
                question,
                context,
                truncation="only_second",
                max_length=self.max_seq_len,
                stride=self.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )
            n_windows = len(enc["input_ids"])
            for w in range(n_windows):
                features.extend(
                    self._window_features(enc, w, n_windows, group, contract_id, context)
                )
        return features

    def _window_features(self, enc, w, n_windows, group, contract_id, context) -> list[dict]:
        seq_ids = enc.sequence_ids(w)
        offsets = enc["offset_mapping"][w]
        ctx = [k for k, s in enumerate(seq_ids) if s == 1]
        if not ctx:
            return []

        hits = []
        for e in group:
            span = None
            if e["answer_text"] is not None:
                start = e["answer_start"]
                span = label_window(offsets, ctx[0], ctx[-1], start, start + len(e["answer_text"]))
            hits.append(
                None
                if span is None
                else (span, e["answer_text"], offsets[span[0]][0], offsets[span[1]][1])
            )
        n_hits = sum(h is not None for h in hits)

        if self.answer_mode == "multi":
            labelled = [h for h in hits if h is not None] or [None]
        elif self.answer_mode == "merged":
            labelled = [next((h for h in hits if h is not None), None)]
        else:
            labelled = hits

        ids = np.asarray(enc["input_ids"][w], dtype=np.int32)
        mask = np.asarray(enc["attention_mask"][w], dtype=np.int32)
        features = []
        for hit in labelled:
            if hit is None:
                (start_tok, end_tok), gold, recon = (0, 0), "", ""
            else:
                (start_tok, end_tok), gold, cs, ce = hit
                recon = context[cs:ce]
            features.append(
                {
                    "input_ids": ids,
                    "attention_mask": mask,
                    "start_positions": int(start_tok),
                    "end_positions": int(end_tok),
                    "contract_id": contract_id,
                    "split": group[0]["split"],
                    "clause_type": group[0]["clause_type"] or "",
                    "window_index": w,
                    "n_windows": n_windows,
                    "is_positive": hit is not None,
                    "n_answers_in_window": n_hits,
                    "_gold": gold,
                    "_recon": recon,
                }
            )
        return features


def balance(features: list[dict], neg_ratio: float, rng: random.Random) -> list[dict]:
    """Keep every positive and `neg_ratio` negatives per positive, sampled uniformly."""
    pos = [i for i, f in enumerate(features) if f["is_positive"]]
    neg = [i for i, f in enumerate(features) if not f["is_positive"]]
    want = min(len(neg), int(round(neg_ratio * len(pos))))
    rng.shuffle(neg)
    keep = pos + neg[:want]
    rng.shuffle(keep)
    return [features[i] for i in keep]


def verify(features: list[dict]) -> dict:
    """Check that each positive's token span reconstructs its gold text."""
    checked = ok = 0
    jaccards: list[float] = []
    failures: list[tuple[str, str]] = []
    for f in features:
        if not f["is_positive"]:
            continue
        checked += 1
        gold, recon = _normalise(f["_gold"]), _normalise(f["_recon"])
        gw, rw = set(gold.split()), set(recon.split())
        jaccards.append(len(gw & rw) / len(gw | rw) if (gw or rw) else 1.0 if not gold else 0.0)
        # Punctuation-only gold answers ("[*]") normalise to nothing; compare them raw.
        if (gold and gold in recon) or (not gold and f["_gold"].strip() in f["_recon"]):
            ok += 1
        elif len(failures) < 5:
            failures.append((f["_gold"][:90], f["_recon"][:90]))
    return {
        "positives_checked": checked,
        "containment_pass_rate": ok / checked if checked else 0.0,
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
        "sample_failures": failures,
    }


def to_dataset(features: list[dict]):
    from datasets import Dataset

    columns = {
        "input_ids": np.stack([f["input_ids"] for f in features]),
        "attention_mask": np.stack([f["attention_mask"] for f in features]),
        "start_positions": np.asarray([f["start_positions"] for f in features], dtype=np.int64),
        "end_positions": np.asarray([f["end_positions"] for f in features], dtype=np.int64),
    }
    for c in META_COLUMNS:
        columns[c] = [f[c] for f in features]
    return Dataset.from_dict(columns)


def _by_contract(examples: list[dict], split: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        if ex["split"] == split:
            grouped[ex["contract_id"]].append(ex)
    return grouped


def probe(
    examples: list[dict], contexts: dict[str, str], builder: FeatureBuilder, split: str, n: int
) -> tuple[float, float]:
    """Features per contract and positive rate, estimated on the first `n` contracts."""
    grouped = _by_contract(examples, split)
    chosen = sorted(grouped)[:n]
    n_feat = n_pos = 0
    for cid in chosen:
        for f in builder(cid, contexts[cid], grouped[cid]):
            n_feat += 1
            n_pos += bool(f["is_positive"])
    if not n_feat:
        return 0.0, 0.0
    return n_feat / len(chosen), n_pos / n_feat


def build_split(
    split: str,
    examples: list[dict],
    contexts: dict[str, str],
    builder: FeatureBuilder,
    neg_ratio: float,
    seed: int,
    prefilter: float | None,
) -> tuple[list[dict], dict]:
    """Featurize every contract of a split and balance the result.

    `prefilter` drops negatives at random while building, to bound memory; it keeps
    enough of them for the exact balancing at the end.
    """
    grouped = _by_contract(examples, split)
    rng = random.Random(seed)
    prefilter_rng = random.Random(seed + 1)
    raw_pos = raw_neg = 0
    kept: list[dict] = []
    start = time.time()

    for i, cid in enumerate(sorted(grouped), 1):
        for f in builder(cid, contexts[cid], grouped[cid]):
            if f["is_positive"]:
                raw_pos += 1
                kept.append(f)
                continue
            raw_neg += 1
            if prefilter is None or prefilter_rng.random() < prefilter:
                kept.append(f)
        if i % 50 == 0 or i == len(grouped):
            print(
                f"  [{split}] {i}/{len(grouped)} contracts, {raw_pos:,} positive, "
                f"{raw_neg:,} negative, {time.time() - start:.0f}s",
                flush=True,
            )

    if len(kept) - raw_pos < round(neg_ratio * raw_pos):
        raise RuntimeError("the negative prefilter kept too few negatives; raise --mem-cap-tokens")

    balanced = balance(kept, neg_ratio, rng)
    n_pos = sum(1 for f in balanced if f["is_positive"])
    raw_total = raw_pos + raw_neg
    stats = {
        "contracts": len(grouped),
        "examples": sum(len(v) for v in grouped.values()),
        "raw_features": raw_total,
        "raw_positive": raw_pos,
        "raw_negative": raw_neg,
        "raw_positive_pct": 100 * raw_pos / raw_total if raw_total else 0.0,
        "prefilter_keep_prob": prefilter,
        "prefilter_dropped": raw_neg - (len(kept) - raw_pos),
        "balanced_features": len(balanced),
        "balanced_positive": n_pos,
        "balanced_positive_pct": 100 * n_pos / len(balanced) if balanced else 0.0,
        "tokens_per_epoch": len(balanced) * builder.max_seq_len,
    }
    return balanced, stats


def _prefilter(
    examples, contexts, builder, split, neg_ratio, mem_cap_tokens, n_probe
) -> float | None:
    per_contract, pos_rate = probe(examples, contexts, builder, split, n_probe)
    projected = per_contract * len(_by_contract(examples, split))
    print(f"  probe: {per_contract:.0f} features/contract, {100 * pos_rate:.2f}% positive")
    if projected * builder.max_seq_len <= mem_cap_tokens or pos_rate == 0:
        return None
    positives = projected * pos_rate
    # Three times the negatives balancing needs, so the exact sample is always possible.
    return min(1.0, 3.0 * neg_ratio * positives / max(projected - positives, 1.0))


def write_sample(features: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for f in features[:SAMPLE_SIZE]:
            record = {
                k: f[k] for k in ("contract_id", "clause_type", "window_index", "is_positive")
            }
            record.update(
                start_positions=f["start_positions"],
                end_positions=f["end_positions"],
                gold=f["_gold"],
                reconstructed=f["_recon"],
            )
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--train-json", type=Path, default=Path("data/raw/train_separate_questions.json")
    )
    parser.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    parser.add_argument("--tokenizer", default="jhu-clsp/ettin-encoder-150m")
    parser.add_argument("--out-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--doc-stride", type=int, default=256, help="token overlap of windows")
    parser.add_argument("--answer-mode", choices=ANSWER_MODES, default="multi")
    parser.add_argument(
        "--neg-ratio", type=float, default=1.0, help="negatives per positive (train)"
    )
    parser.add_argument("--val-neg-ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--mem-cap-tokens", type=int, default=150_000_000)
    parser.add_argument("--probe-contracts", type=int, default=15)
    parser.add_argument("--min-verify", type=float, default=0.98)
    parser.add_argument("--splits-to-build", default="train,val")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    wanted = [s for s in args.splits_to_build.split(",") if s]
    if "test" in wanted:
        raise SystemExit("features are never built for the test split")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    n_windows = assert_sliding_windows(tok, args.max_seq_len, args.doc_stride)
    examples, contexts = load_examples(args.train_json, args.splits, wanted)
    print(f"{len(examples):,} examples from {len(contexts)} contracts")
    builder = FeatureBuilder(tok, args.max_seq_len, args.doc_stride, args.answer_mode)

    tag = (
        f"{args.tokenizer.split('/')[-1]}_w{args.max_seq_len}_s{args.doc_stride}"
        f"_{args.answer_mode}_neg{args.neg_ratio:g}"
    )
    out_root = args.out_dir / tag
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.json"
    previous = json.loads(manifest_path.read_text())["splits"] if manifest_path.exists() else {}

    import tokenizers
    import transformers

    manifest = {
        "tag": tag,
        "tokenizer": args.tokenizer,
        "max_seq_len": args.max_seq_len,
        "doc_stride": args.doc_stride,
        "answer_mode": args.answer_mode,
        "neg_ratio": args.neg_ratio,
        "val_neg_ratio": args.val_neg_ratio,
        "seed": args.seed,
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "sliding_window_check_windows": n_windows,
        "splits": {k: v for k, v in previous.items() if k not in wanted},
    }

    for split in wanted:
        ratio = args.neg_ratio if split == "train" else args.val_neg_ratio
        print(f"\n{split} (negatives per positive: {ratio:g})")
        prefilter = _prefilter(
            examples, contexts, builder, split, ratio, args.mem_cap_tokens, args.probe_contracts
        )
        features, stats = build_split(
            split, examples, contexts, builder, ratio, args.seed, prefilter
        )
        check = verify(features)
        stats["verification"] = check
        stats["positive_by_clause_type"] = dict(
            Counter(f["clause_type"] for f in features if f["is_positive"]).most_common()
        )
        stats["clause_types_with_positives"] = len(stats["positive_by_clause_type"])
        print(
            f"  {stats['raw_features']:,} features ({stats['raw_positive_pct']:.2f}% positive) -> "
            f"{stats['balanced_features']:,} balanced; spans verified "
            f"{100 * check['containment_pass_rate']:.2f}%, "
            f"{stats['clause_types_with_positives']}/41 clause types with a positive"
        )
        if check["positives_checked"] and check["containment_pass_rate"] < args.min_verify:
            raise SystemExit(f"span verification failed: {check['sample_failures']}")

        to_dataset(features).save_to_disk(str(out_root / split))
        manifest["splits"][split] = stats
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        if split == "train":
            write_sample(features, out_root / "sample.jsonl")

    print(f"\nwrote {out_root}")


if __name__ == "__main__":
    main()
