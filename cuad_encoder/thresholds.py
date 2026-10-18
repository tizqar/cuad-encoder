"""Choose two confidence thresholds on validation predictions and score them once on test.

tau_low maximises validation citation F1 (ties go to the higher tau); tau_high is the
smallest tau whose validation citation precision reaches the target.

    python -m cuad_encoder.thresholds --val-preds results/<run>_VAL_ep3_preds.jsonl \
        --test-preds results/<run>_preds.jsonl --out results/<run>_operating_point.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .metrics import RELAXED_CLAUSE_TYPES, GoldDict, PredDict, get_jaccard, is_match

TAU_GRID: tuple[float, ...] = tuple(np.round(np.arange(0.01, 1.0, 0.01), 2))
OUTCOMES = ("correct", "wrong", "false", "omission", "abstain")


def load(preds_path: Path, cuad_json: Path) -> tuple[GoldDict, PredDict, dict[str, str]]:
    from .evaluate import gold_dict, load_eval_set, load_preds

    eval_set = load_eval_set(Path(cuad_json))
    texts = {cid: e["context"] for cid, e in eval_set.items()}
    return gold_dict(eval_set), load_preds(Path(preds_path)), texts


def outcome(golds: list[str], text: str, conf: float, tau: float, relaxed: bool) -> str:
    """Classify the top candidate of one (contract, clause type) key."""
    shown = bool(text) and conf >= tau
    if not golds:
        return "false" if shown else "abstain"
    if not shown:
        return "omission"
    return "correct" if any(is_match(g, text, relaxed) for g in golds) else "wrong"


def _div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def score(gt: GoldDict, preds: PredDict, texts: dict[str, str], tau: float) -> dict:
    """Citation, presence and abstention metrics of the top-1 candidates at one threshold."""
    counts = dict.fromkeys(OUTCOMES, 0)
    jaccards: list[float] = []
    verbatim = 0
    for key, golds in gt.items():
        text, conf = (preds.get(key) or [("", 0.0)])[0]
        result = outcome(golds, text, conf, tau, key[1] in RELAXED_CLAUSE_TYPES)
        counts[result] += 1
        if result in ("correct", "wrong", "false"):
            verbatim += text in texts[key[0]]
        if result == "correct":
            jaccards.append(max(get_jaccard(g, text) for g in golds))

    correct, wrong, false, omission, abstain = (counts[k] for k in OUTCOMES)
    shown = correct + wrong + false
    present, absent = correct + wrong + omission, false + abstain
    p_prec, p_rec = _div(correct + wrong, shown), _div(correct + wrong, present)
    both = p_prec == p_prec and p_rec == p_rec
    return {
        "tau": tau,
        "keys": len(gt),
        "gold_present": present,
        "gold_absent": absent,
        **counts,
        "citation_f1": _div(2 * correct, 2 * correct + wrong + false + omission),
        "citation_precision": _div(correct, shown),
        "citation_recall": _div(correct, present),
        "presence_precision": p_prec,
        "presence_recall": p_rec,
        "presence_f1": _div(2 * p_prec * p_rec, p_prec + p_rec) if both else float("nan"),
        "omission_rate": _div(omission, present),
        "false_citation_rate": _div(false, absent),
        "abstention_accuracy": _div(abstain, absent),
        "mean_jaccard_correct": float(np.mean(jaccards)) if jaccards else float("nan"),
        "verbatim_rate": _div(verbatim, shown),
    }


def choose_tau_low(
    gt: GoldDict, preds: PredDict, texts: dict[str, str], grid: Sequence[float] = TAU_GRID
) -> tuple[float, list[dict]]:
    sweep = [score(gt, preds, texts, t) for t in grid]
    best = max(sweep, key=lambda r: (r["citation_f1"], r["tau"]))
    return best["tau"], sweep


def choose_tau_high(sweep: list[dict], target: float) -> float | None:
    taus = [r["tau"] for r in sweep if r["citation_precision"] >= target]
    return min(taus) if taus else None


def describe(r: dict) -> str:
    return (
        f"tau {r['tau']:.2f} | citation F1 {r['citation_f1']:.3f}  "
        f"precision {r['citation_precision']:.3f}  recall {r['citation_recall']:.3f} | "
        f"presence F1 {r['presence_f1']:.3f} | omission {r['omission_rate']:.3f}  "
        f"false citation {r['false_citation_rate']:.3f} | Jaccard {r['mean_jaccard_correct']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--val-preds", required=True)
    parser.add_argument("--val-json", default="data/raw/val.json")
    parser.add_argument("--test-preds")
    parser.add_argument("--test-json", default="data/raw/test.json")
    parser.add_argument("--precision-target", type=float, default=0.75)
    parser.add_argument("--out")
    args = parser.parse_args()

    gt, preds, texts = load(Path(args.val_preds), Path(args.val_json))
    tau_low, sweep = choose_tau_low(gt, preds, texts)
    tau_high = choose_tau_high(sweep, args.precision_target)
    by_tau = {r["tau"]: r for r in sweep}
    validation = {"tau_low": by_tau[tau_low], "tau_high": by_tau.get(tau_high)}
    print("validation\n  tau_low  " + describe(validation["tau_low"]))
    if tau_high is not None:
        print("  tau_high " + describe(validation["tau_high"]))

    out = {
        "tau_low": tau_low,
        "tau_high": tau_high,
        "precision_target": args.precision_target,
        "criterion": "tau_low = max citation F1 on validation (ties -> higher); "
        "tau_high = smallest tau with validation citation precision >= target",
        "val_preds": args.val_preds,
        "validation": validation,
    }
    if args.test_preds:
        test_gt, test_preds, test_texts = load(Path(args.test_preds), Path(args.test_json))
        test = {"tau_low": score(test_gt, test_preds, test_texts, tau_low)}
        if tau_high is not None:
            test["tau_high"] = score(test_gt, test_preds, test_texts, tau_high)
        print("test")
        for name, r in test.items():
            print(f"  {name:8s} " + describe(r))
        out.update(test_preds=args.test_preds, test=test)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
