"""Train, choose the epoch on validation and score the test split, for one or more seeds.

python -m cuad_encoder.pipeline --features data/features/<tag> --seeds 3407 42 1234 --gpus 2
"""

from __future__ import annotations

import argparse
import codecs
import csv
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .evaluate import load_eval_set

PUBLISHED_AUPR = {"RoBERTa-base": 42.6, "RoBERTa-large": 48.2, "DeBERTa-v2-xlarge": 47.8}


@dataclass
class Config:
    features: str
    model: str = "jhu-clsp/ettin-encoder-150m"
    results: str = "results"
    runs: str = "runs"
    max_seq_len: int = 1024
    doc_stride: int = 256
    max_epochs: int = 3
    select_min_epoch: float = 2
    lr: float = 5e-5
    warmup: float = 0.06
    weight_decay: float = 0.01
    grad_acc: int = 2
    eval_bs: int = 8
    grad_ckpt: bool = False
    n_gpus: int = 1
    device: str | None = None
    val_json: str = "data/raw/val.json"
    test_json: str = "data/raw/test.json"
    keep_checkpoints: bool = False
    extra_train_args: list[str] = field(default_factory=list)

    @property
    def feature_tag(self) -> str:
        return Path(self.features).name

    def run_name(self, seed: int) -> str:
        return f"{self.feature_tag}_seed{seed}_e{self.max_epochs}v"


def _stream(cmd: list[str], log_path: Path | None = None) -> None:
    """Run a command and echo its output as it arrives."""
    print("$", shlex.join(cmd), flush=True)
    # Inside Jupyter an inherited stdout goes to the kernel log, not the cell.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    log = open(log_path, "wb") if log_path else None
    tail: list[str] = []
    try:
        while chunk := proc.stdout.read1(65536):
            if log:
                log.write(chunk)
            text = decoder.decode(chunk)
            sys.stdout.write(text)
            sys.stdout.flush()
            tail = (tail + text.splitlines())[-25:]
    finally:
        if log:
            log.close()
    if code := proc.wait():
        raise SystemExit(f"command failed (exit {code}): {shlex.join(cmd)}\n" + "\n".join(tail))


def train(cfg: Config, seed: int) -> Path:
    run_dir = Path(cfg.runs) / cfg.run_name(seed)
    if (run_dir / "run_summary.json").exists():
        print(f"seed {seed}: already trained in {run_dir}")
        return run_dir
    if cfg.n_gpus > 1:
        launcher = ["accelerate", "launch", "--num_processes", str(cfg.n_gpus)]
    else:
        launcher = [sys.executable]
    cmd = launcher + ["-m", "cuad_encoder.train", "--features", cfg.features, "--model", cfg.model]
    cmd += ["--out", str(run_dir), "--max-epochs", str(cfg.max_epochs), "--final", "last"]
    cmd += ["--lr", str(cfg.lr), "--warmup", str(cfg.warmup)]
    cmd += ["--weight-decay", str(cfg.weight_decay), "--grad-acc", str(cfg.grad_acc)]
    cmd += ["--eval-bs", str(cfg.eval_bs), "--seed", str(seed)]
    cmd += ["--max-seq-len", str(cfg.max_seq_len), "--doc-stride", str(cfg.doc_stride)]
    cmd += ["--grad-ckpt"] if cfg.grad_ckpt else []
    cmd += cfg.extra_train_args
    run_dir.mkdir(parents=True, exist_ok=True)
    _stream(cmd, run_dir / "train.log")
    return run_dir


def choose_epoch(cfg: Config, seed: int) -> dict:
    from .select_epoch import select

    run_dir = Path(cfg.runs) / cfg.run_name(seed)
    selection = run_dir / "selection.json"
    if selection.exists() and (run_dir / "final" / "config.json").exists():
        print(f"seed {seed}: epoch already selected")
        return json.loads(selection.read_text())
    return select(
        run_dir,
        Path(cfg.results),
        cfg.run_name(seed),
        Path(cfg.val_json),
        cfg.max_seq_len,
        cfg.doc_stride,
        n_gpus=cfg.n_gpus,
        device=cfg.device,
        min_epoch=cfg.select_min_epoch,
    )


def _evaluate_cmd(cfg: Config, tag: str) -> list[str]:
    cmd = [sys.executable, "-m", "cuad_encoder.evaluate", "--cuad-json", cfg.test_json]
    cmd += ["--tag", tag, "--out-dir", cfg.results]
    return cmd + (["--device", cfg.device] if cfg.device else [])


def _run_shards(cfg: Config, base: list[str], tag: str, n: int) -> None:
    """One inference process per GPU over disjoint subsets of the contracts."""
    results = Path(cfg.results)
    procs = []
    for k in range(n):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(k)}
        log_path = results / f"{tag}_shard{k}.log"
        log = open(log_path, "w")
        cmd = base + ["--shard", f"{k}/{n}"]
        print(f"GPU {k}: {shlex.join(cmd)}", flush=True)
        procs.append(
            (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log, log_path)
        )
    codes = [proc.wait() for proc, _, _ in procs]
    for _, log, _ in procs:
        log.close()
    for code, (_, _, log_path) in zip(codes, procs, strict=True):
        if code:
            raise SystemExit("test shard failed:\n" + "".join(open(log_path).readlines()[-20:]))

    merged = results / f"{tag}_preds.jsonl"
    with merged.open("w") as out:
        for k in range(n):
            out.write((results / f"{tag}_shard{k}of{n}_preds.jsonl").read_text())
    shutil.copy(
        results / f"{tag}_shard0of{n}_preds.config.json", results / f"{tag}_preds.config.json"
    )
    expected = len(load_eval_set(Path(cfg.test_json)))
    n_lines = sum(1 for line in merged.open() if line.strip())
    if n_lines != expected:
        raise SystemExit(f"expected {expected} contracts after merging shards, got {n_lines}")


def evaluate_test(cfg: Config, model_dir: str, tag: str) -> dict:
    results = Path(cfg.results)
    results.mkdir(parents=True, exist_ok=True)
    report = results / f"{tag}.json"
    if report.exists() and (results / f"{tag}_by_type.csv").exists():
        print(f"{tag}: test already scored")
        return json.loads(report.read_text())
    cmd = _evaluate_cmd(cfg, tag)
    inference = cmd + ["--model", model_dir, "--max-seq-len", str(cfg.max_seq_len)]
    inference += ["--doc-stride", str(cfg.doc_stride)]
    if cfg.n_gpus == 1:
        _stream(inference + ["--curve"])
    else:
        _run_shards(cfg, inference, tag, cfg.n_gpus)
        _stream(cmd + ["--score-only", "--curve"])
    return json.loads(report.read_text())


def seed_row(seed: int, tag: str, selection: dict, report: dict) -> dict:
    """One seed's selected epoch and test scores, in percent."""
    o = report["overall"]
    return {
        "seed": seed,
        "tag": tag,
        "selected_epoch": selection["selected"]["epoch"],
        "val_aupr": selection["selected"]["val_aupr"] * 100,
        "aupr": o["aupr"] * 100,
        "p80": o["prec_at_80_recall"] * 100,
        "p90": o["prec_at_90_recall"] * 100,
        "macro_aupr": report["macro"]["macro_aupr"] * 100,
    }


def run_seed(cfg: Config, seed: int) -> dict:
    start = time.time()
    tag = cfg.run_name(seed)
    print(f"\nseed {seed} -> {tag}", flush=True)
    run_dir = train(cfg, seed)
    selection = choose_epoch(cfg, seed)
    report = evaluate_test(cfg, str(run_dir / "final"), tag)
    if not cfg.keep_checkpoints:
        # final/ is a copy of the selected epoch and selection.json keeps every epoch's score.
        for checkpoint in run_dir.glob("checkpoint-*"):
            shutil.rmtree(checkpoint, ignore_errors=True)
    row = seed_row(seed, tag, selection, report)
    print(
        f"seed {seed}: epoch {row['selected_epoch']:g} (val AUPR {row['val_aupr']:.2f}) -> "
        f"test AUPR {row['aupr']:.2f}  P@80R {row['p80']:.2f}  P@90R {row['p90']:.2f}  "
        f"({(time.time() - start) / 60:.0f} min)",
        flush=True,
    )
    return row


def summarize(rows: list[dict], out_csv: Path) -> dict[str, dict]:
    """Mean, sample standard deviation and range over seeds, written next to the per-seed rows."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    aggregate = {}
    for key in ("aupr", "p80", "p90", "macro_aupr"):
        values = [r[key] for r in rows]
        aggregate[key] = {
            "mean": round(statistics.mean(values), 2),
            "std": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "n": len(values),
        }
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"seeds": rows, "aggregate": aggregate}
    out_csv.with_suffix(".json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'seed':>6s} {'epoch':>6s} {'val':>7s} {'AUPR':>7s} {'P@80R':>7s} {'P@90R':>7s}")
    for r in rows:
        print(
            f"{r['seed']:6d} {r['selected_epoch']:6g} {r['val_aupr']:7.2f} "
            f"{r['aupr']:7.2f} {r['p80']:7.2f} {r['p90']:7.2f}"
        )
    for key, name in (("aupr", "AUPR"), ("p80", "P@80R")):
        a = aggregate[key]
        spread = f"range {a['min']:.2f}-{a['max']:.2f}  n={a['n']}"
        print(f"{name:6s} {a['mean']:.2f} ± {a['std']:.2f}  {spread}")
    a = aggregate["aupr"]
    for name, bar in PUBLISHED_AUPR.items():
        if a["min"] > bar:
            verdict = "every seed above"
        elif a["mean"] > bar:
            verdict = "mean above, not every seed"
        else:
            verdict = "below on average"
        print(f"  vs {name} published ({bar}): {verdict}")
    print(f"wrote {out_csv} and {out_csv.with_suffix('.json')}")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--features", required=True)
    parser.add_argument("--model", default="jhu-clsp/ettin-encoder-150m")
    parser.add_argument("--seeds", type=int, nargs="+", default=[3407])
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--results", default="results")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--keep-checkpoints", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        features=args.features,
        model=args.model,
        results=args.results,
        runs=args.runs,
        grad_ckpt=args.grad_ckpt,
        n_gpus=args.gpus,
        keep_checkpoints=args.keep_checkpoints,
    )
    rows = [run_seed(cfg, seed) for seed in args.seeds]
    summary = Path(cfg.results) / f"{cfg.feature_tag}_e{cfg.max_epochs}v_seeds_summary.csv"
    summarize(rows, summary)


if __name__ == "__main__":
    main()
