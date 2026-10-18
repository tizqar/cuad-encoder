"""Choose a run's epoch by AUPR on the validation contracts and install it as `final/`.

Validation loss is not used: with several gold spans per window it rises while AUPR
still improves.

    python -m cuad_encoder.select_epoch --run runs/<run> --tag <run>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CHECKPOINT_IGNORE = ("optimizer.pt", "scheduler.pt", "rng_state*", "scaler.pt")


def checkpoints(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))


def epoch_of(checkpoint: Path) -> float:
    state = checkpoint / "trainer_state.json"
    return float(json.loads(state.read_text())["epoch"]) if state.exists() else float("nan")


def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return 1
    return max(1, sum(1 for line in out.splitlines() if line.startswith("GPU")))


def _score_checkpoints(
    jobs: list[dict], val_json: Path, results: Path, n_gpus: int, device, decode
):
    """Score each checkpoint on validation, one evaluator process per GPU."""
    pending = [j for j in jobs if not j["report"].exists()]
    while pending:
        wave, pending = pending[:n_gpus], pending[n_gpus:]
        procs = []
        for slot, job in enumerate(wave):
            env = dict(os.environ)
            cmd = [sys.executable, "-m", "cuad_encoder.evaluate", "--cuad-json", str(val_json)]
            cmd += ["--model", job["checkpoint"], "--tag", job["tag"], "--out-dir", str(results)]
            cmd += [f"--{k.replace('_', '-')}={v}" for k, v in decode.items()]
            if device is None:
                env["CUDA_VISIBLE_DEVICES"] = str(slot)
            else:
                cmd += ["--device", device]
            log = open(job["log"], "w")
            procs.append(
                (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log, job)
            )
            print(f"  epoch {job['epoch']:g} <- {job['checkpoint']}", flush=True)
        for proc, log, job in procs:
            code = proc.wait()
            log.close()
            if code:
                tail = "".join(open(job["log"]).readlines()[-15:])
                raise RuntimeError(f"validation of epoch {job['epoch']:g} failed:\n{tail}")


def select(
    run_dir: Path,
    results: Path,
    tag: str,
    val_json: Path = Path("data/raw/val.json"),
    max_seq_len: int = 1024,
    doc_stride: int = 256,
    batch_size: int = 16,
    n_gpus: int | None = None,
    device: str | None = None,
    install_final: bool = True,
    min_epoch: float = 0.0,
) -> dict:
    """Score the checkpoints of a run on validation and keep the best one."""
    run_dir, results = Path(run_dir), Path(results)
    results.mkdir(parents=True, exist_ok=True)
    jobs = []
    for checkpoint in checkpoints(run_dir):
        epoch = epoch_of(checkpoint)
        if epoch < min_epoch:
            continue
        job_tag = f"{tag}_VAL_ep{epoch:g}"
        jobs.append(
            {
                "checkpoint": str(checkpoint),
                "epoch": epoch,
                "tag": job_tag,
                "report": results / f"{job_tag}.json",
                "log": results / f"{job_tag}.log",
            }
        )
    if not jobs:
        raise FileNotFoundError(f"no checkpoints to score under {run_dir}")

    start = time.time()
    decode = {"max_seq_len": max_seq_len, "doc_stride": doc_stride, "batch_size": batch_size}
    _score_checkpoints(jobs, val_json, results, n_gpus or gpu_count(), device, decode)
    print(f"validation scored in {(time.time() - start) / 60:.1f} min")

    rows = []
    for job in jobs:
        report = json.loads(job["report"].read_text())
        o = report["overall"]
        rows.append(
            {
                "epoch": job["epoch"],
                "checkpoint": job["checkpoint"],
                "val_aupr": o["aupr"],
                "val_p80": o["prec_at_80_recall"],
                "val_p90": o["prec_at_90_recall"],
                "val_macro_aupr": report["macro"]["macro_aupr"],
            }
        )
    # Ties go to the higher P@80R, then to the later epoch.
    best = max(rows, key=lambda r: (round(r["val_aupr"], 6), round(r["val_p80"], 6), r["epoch"]))
    for r in rows:
        mark = "  <- selected" if r is best else ""
        scores = f"val AUPR {r['val_aupr'] * 100:.2f}  P@80R {r['val_p80'] * 100:.2f}"
        print(f"  epoch {r['epoch']:g}: {scores}{mark}")

    selection = {
        "criterion": "validation AUPR on CUAD's grid; ties: P@80R, then the later epoch",
        "val_json": str(val_json),
        "rows": rows,
        "selected": best,
    }
    if install_final:
        final = run_dir / "final"
        if final.exists():
            shutil.rmtree(final)
        shutil.copytree(
            best["checkpoint"], final, ignore=shutil.ignore_patterns(*CHECKPOINT_IGNORE)
        )
        selection["final"] = str(final)
    (run_dir / "selection.json").write_text(json.dumps(selection, indent=2))
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--val-json", type=Path, default=Path("data/raw/val.json"))
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--doc-stride", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--device", help="e.g. cpu; by default one GPU per checkpoint")
    parser.add_argument("--min-epoch", type=float, default=0.0)
    parser.add_argument("--no-install", action="store_true")
    args = parser.parse_args()
    select(
        args.run,
        args.results,
        args.tag,
        args.val_json,
        args.max_seq_len,
        args.doc_stride,
        args.batch_size,
        args.gpus,
        args.device,
        not args.no_install,
        args.min_epoch,
    )


if __name__ == "__main__":
    main()
