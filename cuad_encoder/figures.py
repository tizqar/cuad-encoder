"""Figures and comparison tables from saved evaluation reports.

    python -m cuad_encoder.figures mean --tag <tag>_SEEDMEAN results/<seed run>.json ...
    python -m cuad_encoder.figures draw --out results/figures/<name> \
        --system "<label>=results/<run>_by_type.csv" --preds "<label>=results/<run>_preds.jsonl"
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from functools import reduce
from operator import getitem
from pathlib import Path

from .clause_types import clause_type_from_question

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL = "#8a8985"
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

# Hendrycks et al. 2021, Table 2: AUPR and P@80R on the official test split.
PUBLISHED: dict[str, tuple[float, float]] = {
    "RoBERTa-base (125M), published": (42.6, 31.1),
    "RoBERTa-large (355M), published": (48.2, 38.1),
    "DeBERTa-v2-xlarge (900M), published": (47.8, 44.0),
}

POSITIONAL = {"Document Name", "Parties", "Agreement Date", "Effective Date"}
METRICS = ("aupr", "prec_at_80_recall", "prec_at_90_recall")


def _percent(value: str | None) -> float | None:
    return float(value) * 100 if value not in (None, "") else None


def load_by_type(csv_path: Path) -> dict:
    """Read a `_by_type.csv` into {"types", "overall", "macro"}, values in percent."""
    out: dict = {"types": {}, "overall": None, "macro": None}
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            record = {
                "n_gold": int(row["n_gold"]) if row["n_gold"] else None,
                "aupr": _percent(row["aupr"]),
                "p80": _percent(row["prec_at_80_recall"]),
                "p90": _percent(row["prec_at_90_recall"]),
                "excluded": bool(row["excluded"]),
            }
            name = row["clause_type"]
            if name == "__overall__":
                out["overall"] = record
            elif name == "__macro__":
                out["macro"] = record
            else:
                out["types"][name] = record
    return out


def mean_report(reports: list[dict]) -> dict:
    """Average evaluation reports of several seeds into one report of the same shape."""

    def mean(*keys: str) -> float:
        return statistics.mean(reduce(getitem, keys, r) for r in reports)

    first = reports[0]
    by_type = {
        ct: {"n_gold": m["n_gold"], **{k: mean("by_clause_type", ct, k) for k in METRICS}}
        for ct, m in first["by_clause_type"].items()
    }
    return {
        "overall": {k: mean("overall", k) for k in first["overall"]},
        "macro": {k: mean("macro", k) for k in first["macro"]},
        "by_clause_type": by_type,
        "excluded_clause_types": first["excluded_clause_types"],
        "n_gold_answers": first["n_gold_answers"],
        "n_keys": first["n_keys"],
        "config": {"models": [r["config"]["model"] for r in reports]},
    }


def sibling_share(cuad_json: Path) -> dict[str, tuple[int, float]]:
    """Per clause type: gold answers, and the share that are a 2nd+ span of their key."""
    total: dict[str, int] = defaultdict(int)
    extra: dict[str, int] = defaultdict(int)
    for article in json.loads(Path(cuad_json).read_text(encoding="utf-8"))["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                n = len(qa.get("answers") or [])
                if n:
                    ct = clause_type_from_question(qa["question"])
                    total[ct] += n
                    extra[ct] += n - 1
    return {ct: (total[ct], extra[ct] / total[ct]) for ct in total}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT2,
            "axes.titlecolor": TEXT,
            "xtick.color": TEXT2,
            "ytick.color": TEXT2,
            "text.color": TEXT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )
    return plt


def _colors(labels: list[str]) -> dict[str, str]:
    """One fixed hue per system, by position, so a system keeps its colour across figures."""
    if len(labels) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} systems per figure")
    return {label: SERIES[i] for i, label in enumerate(labels)}


def _save(fig, plt, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def summary_rows(runs: dict[str, dict]) -> list[dict]:
    rows = []
    for label, d in runs.items():
        o, m = d["overall"], d["macro"]
        rows.append(
            {
                "system": label,
                "aupr": round(o["aupr"], 2),
                "p80": round(o["p80"], 2),
                "p90": round(o["p90"], 2),
                "macro_aupr": round(m["aupr"], 2) if m and m["aupr"] is not None else "",
                "types_reported": len(d["types"]),
            }
        )
    for label, (aupr, p80) in PUBLISHED.items():
        rows.append(
            {
                "system": label,
                "aupr": aupr,
                "p80": p80,
                "p90": "",
                "macro_aupr": "",
                "types_reported": "",
            }
        )
    return rows


def _rounded(record: dict | None, key: str) -> float | str:
    return round(record[key], 2) if record and record[key] is not None else ""


def per_type_rows(runs: dict[str, dict], cuad_json: Path | None = None) -> list[dict]:
    labels = list(runs)

    def n_gold(ct: str) -> int:
        return max((d["types"].get(ct) or {}).get("n_gold") or 0 for d in runs.values())

    types = sorted({ct for d in runs.values() for ct in d["types"]}, key=lambda ct: -n_gold(ct))
    share = sibling_share(cuad_json) if cuad_json else {}
    rows = []
    for ct in types:
        row = {"clause_type": ct, "positional": ct in POSITIONAL, "n_gold": n_gold(ct)}
        row["sibling_share"] = round(share[ct][1], 3) if ct in share else ""
        for label in labels:
            record = runs[label]["types"].get(ct)
            row[f"aupr[{label}]"] = _rounded(record, "aupr")
            row[f"p80[{label}]"] = _rounded(record, "p80")
        if len(labels) >= 2:
            a, b = (runs[label]["types"].get(ct) for label in labels[:2])
            delta = (
                a["aupr"] - b["aupr"] if a and b and None not in (a["aupr"], b["aupr"]) else None
            )
            row[f"delta_aupr[{labels[0]} - {labels[1]}]"] = (
                round(delta, 2) if delta is not None else ""
            )
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def fig_run_comparison(runs: dict[str, dict], out: Path) -> Path:
    """Overall AUPR and P@80R per system, with the published results as reference lines."""
    plt = _style()
    labels = list(runs)
    colors = _colors(labels)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(labels) + 2.6), sharey=True)
    for ax, key, title in zip(
        axes, ("aupr", "p80"), ("AUPR (%)", "Precision @ 80% recall (%)"), strict=True
    ):
        values = [runs[label]["overall"][key] for label in labels]
        y = range(len(labels))
        ax.barh(
            y,
            values,
            height=0.55,
            color=[colors[label] for label in labels],
            edgecolor=SURFACE,
            linewidth=2,
        )
        for yi, v in zip(y, values, strict=True):
            ax.text(v + 0.6, yi, f"{v:.2f}", va="center", ha="left", fontsize=9, color=TEXT)
        column = 0 if key == "aupr" else 1
        refs = sorted((pub[column], name.split(",")[0]) for name, pub in PUBLISHED.items())
        for i, (ref, short) in enumerate(refs):
            ax.axvline(ref, color=NEUTRAL, linewidth=1, linestyle=(0, (3, 3)), zorder=0)
            # Three label heights, so close references (47.8 and 48.2) do not overlap.
            ax.annotate(
                f"{short}  {ref:.1f}",
                xy=(ref, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, 4 + 11 * (i % 3)),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=TEXT2,
            )
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels)
        ax.set_xlim(0, max(values + [ref for ref, _ in refs]) * 1.18)
        ax.set_title(title, pad=42)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    fig.text(
        0.01,
        -0.03,
        "CUAD official test split, 102 contracts, CUAD's threshold grid. "
        "Dashed: Hendrycks et al. 2021, Table 2.",
        ha="left",
        fontsize=9,
        color=TEXT2,
    )
    return _save(fig, plt, out)


def _connect(ax, runs: dict[str, dict], labels: list[str], types: list[str], position) -> None:
    """Grey connector between the first two systems for every clause type."""
    if len(labels) < 2:
        return
    for ct in types:
        a = runs[labels[0]]["types"][ct]["aupr"]
        b = (runs[labels[1]]["types"].get(ct) or {}).get("aupr")
        if a is not None and b is not None:
            xs, ys = position(ct, b, a)
            ax.plot(xs, ys, color=GRID, linewidth=2, zorder=1)


def fig_per_type_dots(runs: dict[str, dict], out: Path) -> Path:
    """Per-clause-type AUPR, one row per type sorted by gold count, one dot per system."""
    plt = _style()
    labels = list(runs)
    colors = _colors(labels)
    first = runs[labels[0]]["types"]
    types = sorted(
        (ct for ct, r in first.items() if not r["excluded"]),
        key=lambda ct: -(first[ct]["n_gold"] or 0),
    )
    row = {ct: i for i, ct in enumerate(types)}
    fig, ax = plt.subplots(figsize=(10, 0.27 * len(types) + 1.8))
    _connect(ax, runs, labels, types, lambda ct, b, a: ([b, a], [row[ct]] * 2))
    for label in labels:
        points = [
            (r["aupr"], row[ct])
            for ct in types
            if (r := runs[label]["types"].get(ct)) and r["aupr"] is not None
        ]
        ax.scatter(
            *zip(*points, strict=True),
            s=46,
            color=colors[label],
            edgecolor=SURFACE,
            linewidth=1.2,
            zorder=3,
            label=label,
        )
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(
        [f"{ct}{'  *' if ct in POSITIONAL else ''}   ({first[ct]['n_gold']})" for ct in types]
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(
        "AUPR (%)   --   rows sorted by gold answers (in parentheses); * positional clause type"
    )
    ax.set_title("Per-clause-type AUPR")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    return _save(fig, plt, out)


def fig_sibling_share(
    runs: dict[str, dict], cuad_json: Path, out: Path, min_gold: int = 10
) -> Path:
    """AUPR against the share of a type's gold answers that are a 2nd+ span of their key."""
    plt = _style()
    labels = list(runs)[:3]
    colors = _colors(labels)
    share = sibling_share(cuad_json)
    first = runs[labels[0]]["types"]
    types = [
        ct
        for ct, r in first.items()
        if not r["excluded"] and (r["n_gold"] or 0) >= min_gold and ct in share
    ]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    _connect(ax, runs, labels, types, lambda ct, b, a: ([share[ct][1] * 100] * 2, [b, a]))
    for label in labels:
        records = [
            (ct, r) for ct in types if (r := runs[label]["types"].get(ct)) and r["aupr"] is not None
        ]
        ax.scatter(
            [share[ct][1] * 100 for ct, _ in records],
            [r["aupr"] for _, r in records],
            s=[18 + r["n_gold"] * 0.8 for _, r in records],
            color=colors[label],
            alpha=0.85,
            edgecolor=SURFACE,
            linewidth=1.2,
            zorder=3,
            label=label,
        )
    heavy = sorted(types, key=lambda ct: -(first[ct]["n_gold"] or 0))[:7]
    for i, ct in enumerate(heavy):
        ax.annotate(
            ct,
            (share[ct][1] * 100, first[ct]["aupr"]),
            xytext=(7, 5 if i % 2 == 0 else -11),
            textcoords="offset points",
            fontsize=8,
            color=TEXT2,
        )
    ax.set_xlim(-2, 102)
    ax.set_ylim(0, 100)
    ax.set_xlabel("gold answers that are a 2nd+ span of their (contract, clause type)  (%)")
    ax.set_ylabel("AUPR (%)")
    ax.set_title(f"Multi-span clause types vs. AUPR  (marker area = gold answers, >= {min_gold})")
    ax.legend(loc="lower left", markerscale=0.8)
    return _save(fig, plt, out)


def fig_pr_curves(preds_files: dict[str, Path], cuad_json: Path, out: Path) -> Path:
    """Precision-recall curves on the dense grid, with the published P@80R points."""
    from .evaluate import gold_dict, load_eval_set, load_preds
    from .metrics import DENSE_THRESHOLDS, get_precisions_recalls, process_precisions

    plt = _style()
    labels = list(preds_files)
    colors = _colors(labels)
    gt = gold_dict(load_eval_set(Path(cuad_json)))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    rows = []
    for label in labels:
        preds = load_preds(Path(preds_files[label]))
        for key in gt:
            preds.setdefault(key, [])
        p, r, confs = get_precisions_recalls(gt, preds, thresholds=DENSE_THRESHOLDS)
        envelope = process_precisions(p)
        ax.plot(
            [x * 100 for x in r],
            [v * 100 for v in envelope],
            color=colors[label],
            linewidth=2,
            label=label,
        )
        rows += [
            {
                "system": label,
                "conf": confs[i - 1] if i else 1.0,
                "precision": p[i],
                "envelope": envelope[i],
                "recall": r[i],
            }
            for i in range(len(p))
        ]
    for name, (_, p80) in PUBLISHED.items():
        ax.scatter([80], [p80], s=60, color=NEUTRAL, edgecolor=SURFACE, linewidth=1.2, zorder=4)
        ax.annotate(
            name.split(",")[0],
            (80, p80),
            xytext=(7, -3),
            textcoords="offset points",
            fontsize=8,
            color=TEXT2,
        )
    ax.axvline(80, color=GRID, linewidth=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("recall (%)")
    ax.set_ylabel("precision (%, envelope)")
    ax.set_title("Precision-recall on CUAD's official test split  (grey: published P@80%R)")
    ax.legend(loc="upper right")
    write_csv(rows, out.with_suffix(".csv"))
    return _save(fig, plt, out)


def fig_training_curves(log_history: list[dict], out: Path) -> Path:
    """Training and validation loss, and the validation proxies, per epoch."""
    plt = _style()
    train = [(e["epoch"], e["loss"]) for e in log_history if "loss" in e and "eval_loss" not in e]
    evals = [e for e in log_history if "eval_loss" in e]
    epochs = [e["epoch"] for e in evals]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    if train:
        ax1.plot(
            *zip(*train, strict=True),
            color=SERIES[1],
            linewidth=1.5,
            label="train loss (per log step)",
        )
    if evals:
        losses = [e["eval_loss"] for e in evals]
        ax1.plot(
            epochs,
            losses,
            color=SERIES[0],
            linewidth=2,
            marker="o",
            markersize=6,
            label="validation loss",
        )
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy")
    ax1.set_title("Loss")
    ax1.legend()
    proxies = [
        ("eval_pos_top5", "gold span in top-5 start x top-5 end (positives)"),
        ("eval_answerability_ap", "answerability AP (present vs absent)"),
        ("eval_null_accuracy", "present/absent decision accuracy"),
        ("eval_pos_exact", "exact top-1 span (positives)"),
    ]
    for i, (key, name) in enumerate(proxies):
        if evals and key in evals[0]:
            values = [e[key] for e in evals]
            ax2.plot(
                epochs, values, color=SERIES[i], linewidth=2, marker="o", markersize=6, label=name
            )
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("epoch")
    ax2.set_title("Validation proxies")
    ax2.legend(loc="lower right")
    return _save(fig, plt, out)


def draw(
    out_dir: Path,
    systems: dict[str, Path],
    preds: dict[str, Path],
    cuad_json: Path,
    log_history: Path | None = None,
) -> list[Path]:
    """Every figure and table for the given systems; the first system gets the first colour."""
    runs = {label: load_by_type(path) for label, path in systems.items()}
    first_two = dict(list(runs.items())[:2])
    made = [
        write_csv(summary_rows(runs), out_dir / "summary.csv"),
        write_csv(per_type_rows(runs, cuad_json), out_dir / "per_type.csv"),
        fig_run_comparison(runs, out_dir / "run_comparison.png"),
        fig_per_type_dots(first_two, out_dir / "per_type_aupr.png"),
        fig_sibling_share(runs, cuad_json, out_dir / "sibling_share_vs_aupr.png"),
    ]
    if preds:
        made.append(fig_pr_curves(preds, cuad_json, out_dir / "pr_curves.png"))
    if log_history:
        made.append(
            fig_training_curves(
                json.loads(log_history.read_text()), out_dir / "training_curves.png"
            )
        )
    for path in made:
        print("wrote", path)
    return made


def _pairs(items: list[str]) -> dict[str, Path]:
    pairs = {}
    for item in items:
        label, sep, path = item.partition("=")
        if not sep:
            raise SystemExit(f"expected LABEL=PATH, got {item!r}")
        pairs[label] = Path(path)
    return pairs


def main() -> None:
    from .evaluate import write_results

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    mean = commands.add_parser("mean", help="average the reports of several seeds")
    mean.add_argument("reports", type=Path, nargs="+")
    mean.add_argument("--tag", required=True)
    mean.add_argument("--out-dir", type=Path, default=Path("results"))
    figures = commands.add_parser("draw", help="draw figures for one or more systems")
    figures.add_argument("--out", type=Path, required=True)
    figures.add_argument("--system", action="append", required=True, metavar="LABEL=BY_TYPE_CSV")
    figures.add_argument("--preds", action="append", default=[], metavar="LABEL=PREDS_JSONL")
    figures.add_argument("--log-history", type=Path)
    figures.add_argument("--cuad-json", type=Path, default=Path("data/raw/test.json"))
    args = parser.parse_args()

    if args.command == "mean":
        reports = [json.loads(p.read_text()) for p in args.reports]
        write_results(mean_report(reports), args.out_dir, args.tag)
    else:
        draw(args.out, _pairs(args.system), _pairs(args.preds), args.cuad_json, args.log_history)


if __name__ == "__main__":
    main()
