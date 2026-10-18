"""CUAD's evaluation metric: AUPR and precision at 80% and 90% recall.

Ported from TheAtticusProject/cuad `evaluate.py`. The only change: keys are
(contract, clause type) tuples and clause types are compared exactly, instead of the
original substring test on a string key.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

Key = tuple[str, str]
GoldDict = dict[Key, list[str]]
PredDict = dict[Key, list[tuple[str, float]]]

IOU_THRESH = 0.5

# No gold answers in the official test split, so excluded from the macro average.
EXCLUDED_CLAUSE_TYPES = ("Price Restrictions",)

# Annotated as many short fragments, so a gold span contained in the prediction matches.
RELAXED_CLAUSE_TYPES = ("Parties",)

CUAD_THRESHOLDS = list(np.arange(0.99, 0, -0.01)) + [0.001, 0.0]
FINE_THRESHOLDS = list(np.arange(0.99, 0, -0.01)) + [
    0.008,
    0.006,
    0.005,
    0.004,
    0.003,
    0.002,
    0.001,
    0.0005,
    0.0002,
    0.0001,
    0.0,
]
DENSE_THRESHOLDS = (
    list(np.arange(0.99, 0.1, -0.01))
    + list(np.arange(0.1, 0.001, -0.001))
    + [0.0005, 0.0002, 0.0001, 0.0]
)
# CUAD's grid gives the published-comparable numbers; the denser ones only refine the curve.
GRIDS = {"cuad": CUAD_THRESHOLDS, "fine": FINE_THRESHOLDS, "dense": DENSE_THRESHOLDS}


def get_jaccard(gt: str, pred: str) -> float:
    """Word-set Jaccard overlap, exactly as in CUAD."""
    for token in (".", ",", ";", ":"):
        gt = gt.replace(token, "")
        pred = pred.replace(token, "")
    gt = gt.lower().replace("/", " ")
    pred = pred.lower().replace("/", " ")
    gt_words = set(gt.split(" "))
    pred_words = set(pred.split(" "))
    return len(gt_words & pred_words) / len(gt_words | pred_words)


def is_match(gold: str, pred: str, relaxed: bool = False) -> bool:
    """Whether a prediction matches a gold span."""
    if get_jaccard(gold, pred) >= IOU_THRESH:
        return True
    return relaxed and gold in pred


def compute_precision_recall(
    gt_dict: GoldDict,
    preds_dict: dict[Key, list[str]],
    clause_type: str | None = None,
) -> tuple[float, float]:
    """Precision and recall of thresholded predictions; NaN where undefined."""
    tp = fp = fn = 0
    for key, answers in gt_dict.items():
        if clause_type is not None and key[1] != clause_type:
            continue
        relaxed = key[1] in RELAXED_CLAUSE_TYPES
        preds = preds_dict.get(key, [])
        if not answers:
            fp += len(preds)
            continue
        for answer in answers:
            if any(is_match(answer, p, relaxed) for p in preds):
                tp += 1
            else:
                fn += 1
        for pred in preds:
            if not any(is_match(a, pred, relaxed) for a in answers):
                fp += 1

    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan
    return precision, recall


def threshold_preds(preds_dict: PredDict, conf_thresh: float) -> dict[Key, list[str]]:
    """Predictions at or above a confidence threshold, empty spans dropped."""
    return {
        key: [text for text, conf in preds if conf >= conf_thresh and text.strip()]
        for key, preds in preds_dict.items()
    }


def process_precisions(precisions: Sequence[float]) -> list[float]:
    """Right-running maximum of the precisions (the precision envelope)."""
    best = list(precisions)[::-1]
    for i in range(1, len(best)):
        best[i] = max(best[i - 1], best[i])
    return best[::-1]


def get_precisions_recalls(
    gt_dict: GoldDict,
    preds_dict: PredDict,
    clause_type: str | None = None,
    thresholds: Sequence[float] | None = None,
) -> tuple[list[float], list[float], list[float]]:
    """Precision-recall curve over a threshold sweep, starting at (1, 0)."""
    precisions: list[float] = [1.0]
    recalls: list[float] = [0.0]
    confs: list[float] = []
    for conf in CUAD_THRESHOLDS if thresholds is None else thresholds:
        prec, rec = compute_precision_recall(
            gt_dict, threshold_preds(preds_dict, conf), clause_type
        )
        precisions.append(prec)
        recalls.append(rec)
        confs.append(float(conf))
    return precisions, recalls, confs


def get_aupr(precisions: Sequence[float], recalls: Sequence[float]) -> float:
    """Area under the precision envelope; 0 when recall is undefined."""
    from sklearn import metrics

    processed = process_precisions(precisions)
    if np.any(np.isnan(np.asarray(recalls, dtype=float))):
        return 0.0
    aupr = metrics.auc(recalls, processed)
    return 0.0 if np.isnan(aupr) else float(aupr)


def get_prec_at_recall(
    precisions: Sequence[float], recalls: Sequence[float], recall_thresh: float = 0.8
) -> float:
    """Envelope precision at the first threshold whose recall reaches `recall_thresh`."""
    processed = process_precisions(precisions)
    for prec, recall in zip(processed, recalls, strict=True):
        if not np.isnan(recall) and recall >= recall_thresh:
            return 0.0 if np.isnan(prec) else float(prec)
    return 0.0


def evaluate(
    gt_dict: GoldDict,
    preds_dict: PredDict,
    clause_types: Iterable[str] | None = None,
    thresholds: Sequence[float] | None = None,
) -> dict:
    """Overall, per-clause-type and macro AUPR, P@80R and P@90R."""
    if clause_types is None:
        clause_types = sorted({k[1] for k in gt_dict})
    clause_types = list(clause_types)

    p, r, _ = get_precisions_recalls(gt_dict, preds_dict, thresholds=thresholds)
    overall = {
        "aupr": get_aupr(p, r),
        "prec_at_80_recall": get_prec_at_recall(p, r, 0.8),
        "prec_at_90_recall": get_prec_at_recall(p, r, 0.9),
    }

    by_type: dict[str, dict] = {}
    for ct in clause_types:
        n_gold = sum(len(v) for k, v in gt_dict.items() if k[1] == ct)
        cp, cr, _ = get_precisions_recalls(
            gt_dict, preds_dict, clause_type=ct, thresholds=thresholds
        )
        by_type[ct] = {
            "n_gold": n_gold,
            "aupr": get_aupr(cp, cr),
            "prec_at_80_recall": get_prec_at_recall(cp, cr, 0.8),
            "prec_at_90_recall": get_prec_at_recall(cp, cr, 0.9),
        }

    scored = [
        ct for ct in clause_types if ct not in EXCLUDED_CLAUSE_TYPES and by_type[ct]["n_gold"] > 0
    ]
    macro = {
        f"macro_{m}": float(np.mean([by_type[ct][m] for ct in scored])) if scored else 0.0
        for m in ("aupr", "prec_at_80_recall", "prec_at_90_recall")
    }
    return {
        "overall": overall,
        "macro": macro,
        "by_clause_type": by_type,
        "excluded_clause_types": [ct for ct in clause_types if ct not in scored],
        "n_gold_answers": sum(len(v) for v in gt_dict.values()),
        "n_keys": len(gt_dict),
    }
