"""Run a QA checkpoint over CUAD contracts and score it with CUAD's metric.

python -m cuad_encoder.evaluate --model runs/<run>/final --tag <run>
python -m cuad_encoder.evaluate --tag <run> --score-only --grid dense
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from .clause_types import clause_type_from_question
from .metrics import (
    GRIDS,
    GoldDict,
    PredDict,
    evaluate,
    get_precisions_recalls,
    process_precisions,
    threshold_preds,
)
from .qa import QAModel
from .windowing import assert_sliding_windows

# Part of the predictions' fingerprint; change it whenever the postprocessing changes.
POSTPROCESSING_VERSION = 3

DECODE_KEYS = ("max_seq_len", "doc_stride", "max_answer_len", "top_k")
CURVE_ROWS = (0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.0)


def load_eval_set(cuad_json: Path, limit: int | None = None) -> dict[str, dict]:
    """Contracts of a CUAD-format file: {title: {context, questions, gold}}."""
    data = json.loads(Path(cuad_json).read_text(encoding="utf-8"))["data"]
    contracts: dict[str, dict] = {}
    for article in data:
        entry = contracts.setdefault(
            article["title"],
            {"context": article["paragraphs"][0]["context"], "questions": {}, "gold": {}},
        )
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                clause_type = clause_type_from_question(qa["question"])
                if clause_type is None:
                    continue
                entry["questions"][clause_type] = qa["question"]
                entry["gold"][clause_type] = [a["text"] for a in qa.get("answers") or []]
    if limit is not None:
        contracts = {t: contracts[t] for t in sorted(contracts)[:limit]}
    return contracts


def gold_dict(eval_set: dict[str, dict]) -> GoldDict:
    return {(cid, ct): spans for cid, e in eval_set.items() for ct, spans in e["gold"].items()}


def run_inference(
    eval_set: dict[str, dict], qa: QAModel, preds_path: Path, resume: bool = True, **decode
) -> None:
    """Append predictions to a JSONL file, one line per contract, resuming if possible."""
    stamp = {"impl": POSTPROCESSING_VERSION, **decode}
    stamp_path = preds_path.with_suffix(".config.json")

    done: set[str] = set()
    if resume and preds_path.exists():
        saved = json.loads(stamp_path.read_text()) if stamp_path.exists() else {}
        saved.pop("model", None)
        if saved == stamp:
            done = {json.loads(line)["contract_id"] for line in preds_path.open() if line.strip()}
            print(f"resuming: {len(done)} contract(s) already predicted")
        else:
            print(f"discarding {preds_path.name}: written with other settings ({saved})")
            preds_path.unlink()
    elif preds_path.exists():
        preds_path.unlink()
    stamp_path.write_text(json.dumps({**stamp, "model": qa.model_id}, sort_keys=True))

    todo = [c for c in sorted(eval_set) if c not in done]
    start = time.time()
    with preds_path.open("a", encoding="utf-8") as fh:
        for i, cid in enumerate(todo, 1):
            contract = eval_set[cid]
            preds = {
                ct: qa(question, contract["context"], **decode)
                for ct, question in contract["questions"].items()
            }
            fh.write(json.dumps({"contract_id": cid, "preds": preds}) + "\n")
            fh.flush()
            per_contract = (time.time() - start) / i
            print(
                f"  [{i}/{len(todo)}] {cid[:60]}  ({per_contract:.1f}s/contract, "
                f"ETA {per_contract * (len(todo) - i) / 60:.1f}m)",
                flush=True,
            )


def load_preds(preds_path: Path) -> PredDict:
    preds: PredDict = {}
    for line in Path(preds_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            for ct, spans in record["preds"].items():
                preds[(record["contract_id"], ct)] = [(t, float(c)) for t, c in spans]
    return preds


def print_pr_curve(gt: GoldDict, preds: PredDict) -> None:
    precisions, recalls, confs = get_precisions_recalls(gt, preds)
    envelope = process_precisions(precisions)
    print(f"\n{'conf':>6s} {'precision':>10s} {'envelope':>9s} {'recall':>8s} {'#preds':>8s}")
    for i, conf in enumerate(confs):
        if round(conf, 4) not in CURVE_ROWS:
            continue
        n_preds = sum(len(v) for v in threshold_preds(preds, conf).values())
        p = precisions[i + 1]
        shown = f"{p * 100:10.2f}" if p == p else "       nan"
        envelope_at, recall_at = envelope[i + 1] * 100, recalls[i + 1] * 100
        print(f"{conf:6.2f} {shown} {envelope_at:9.2f} {recall_at:8.2f} {n_preds:8d}")
    print(f"max recall: {max(r for r in recalls if r == r) * 100:.2f}%")


def write_results(report: dict, out_dir: Path, tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    with (out_dir / f"{tag}_by_type.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["clause_type", "n_gold", "aupr", "prec_at_80_recall", "prec_at_90_recall", "excluded"]
        )
        excluded = set(report["excluded_clause_types"])
        for ct, m in sorted(report["by_clause_type"].items()):
            writer.writerow(
                [
                    ct,
                    m["n_gold"],
                    f"{m['aupr']:.4f}",
                    f"{m['prec_at_80_recall']:.4f}",
                    f"{m['prec_at_90_recall']:.4f}",
                    "yes" if ct in excluded else "",
                ]
            )
        o, m = report["overall"], report["macro"]
        writer.writerow(
            [
                "__overall__",
                report["n_gold_answers"],
                f"{o['aupr']:.4f}",
                f"{o['prec_at_80_recall']:.4f}",
                f"{o['prec_at_90_recall']:.4f}",
                "",
            ]
        )
        writer.writerow(
            [
                "__macro__",
                "",
                f"{m['macro_aupr']:.4f}",
                f"{m['macro_prec_at_80_recall']:.4f}",
                f"{m['macro_prec_at_90_recall']:.4f}",
                "",
            ]
        )
    print(f"wrote {out_dir / (tag + '.json')} and {out_dir / (tag + '_by_type.csv')}")


def _shard(eval_set: dict[str, dict], spec: str) -> dict[str, dict]:
    k, n = (int(x) for x in spec.split("/"))
    if not 0 <= k < n:
        raise SystemExit(f"--shard must be K/N with 0 <= K < N, got {spec!r}")
    return {c: v for i, (c, v) in enumerate(sorted(eval_set.items())) if i % n == k}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cuad-json", type=Path, default=Path("data/raw/test.json"))
    parser.add_argument(
        "--model", help="checkpoint directory or hub id (not needed with --score-only)"
    )
    parser.add_argument("--tag", required=True, help="basename of the output files")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--limit", type=int, help="only the first N contracts")
    parser.add_argument("--shard", metavar="K/N", help="only contracts with index %% N == K")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--doc-stride", type=int, default=256)
    parser.add_argument("--max-answer-len", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--grid", choices=GRIDS, default="cuad")
    parser.add_argument("--curve", action="store_true", help="print the precision-recall curve")
    args = parser.parse_args()

    eval_set = load_eval_set(args.cuad_json, args.limit)
    tag = args.tag
    if args.shard:
        if args.score_only:
            raise SystemExit("--shard is for inference; score the merged predictions instead")
        eval_set = _shard(eval_set, args.shard)
        tag += "_shard" + args.shard.replace("/", "of")
    preds_path = args.out_dir / f"{tag}_preds.jsonl"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.cuad_json.stem}: {len(eval_set)} contracts")

    if not args.score_only:
        if not args.model:
            raise SystemExit("--model is required unless --score-only")
        qa = QAModel(args.model, args.device, args.batch_size, fp32=args.fp32)
        print(f"loaded {args.model} on {qa.device} ({qa.dtype})")
        assert_sliding_windows(qa.tok, args.max_seq_len, args.doc_stride)
        decode = {k: getattr(args, k) for k in DECODE_KEYS}
        run_inference(eval_set, qa, preds_path, resume=not args.no_resume, **decode)
    if args.shard:
        return

    # The stamp records the settings the predictions were made with, also under --score-only.
    stamp_path = preds_path.with_suffix(".config.json")
    stamp = json.loads(stamp_path.read_text()) if stamp_path.exists() else {}
    decode = {k: stamp.get(k, getattr(args, k)) for k in DECODE_KEYS}
    gt = gold_dict(eval_set)
    preds = load_preds(preds_path)
    for key in gt:
        preds.setdefault(key, [])
    if args.curve:
        print_pr_curve(gt, preds)

    report = evaluate(gt, preds, thresholds=GRIDS[args.grid])
    report["config"] = {
        "model": stamp.get("model", args.model),
        "split": args.cuad_json.stem,
        "n_contracts": len(eval_set),
        **decode,
        "threshold_grid": args.grid,
    }
    o, macro = report["overall"], report["macro"]["macro_aupr"]
    print(
        f"\nAUPR {o['aupr'] * 100:.2f}  P@80R {o['prec_at_80_recall'] * 100:.2f}  "
        f"P@90R {o['prec_at_90_recall'] * 100:.2f}  macro AUPR {macro * 100:.2f}"
    )
    write_results(report, args.out_dir, tag if args.grid == "cuad" else f"{tag}_{args.grid}")


if __name__ == "__main__":
    main()
