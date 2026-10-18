"""Fine-tune an extractive QA encoder on features from `cuad_encoder.features`.

    python -m cuad_encoder.train --features data/features/<tag> --out runs/<run>
    accelerate launch --num_processes 2 -m cuad_encoder.train --features ... --out ...

Run under `accelerate launch` on several GPUs: Trainer inside a notebook falls back to
`nn.DataParallel`, which was slower on two GPUs than on one.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

FEATURE_COLUMNS = ("input_ids", "attention_mask", "start_positions", "end_positions")


def collate(features):
    import torch

    return {k: torch.stack([f[k] for f in features]).long() for k in FEATURE_COLUMNS}


def preprocess_logits_for_metrics(logits, labels):
    """Reduce each example to its top-5 starts and ends, null score and best span score."""
    import torch

    s, e = logits[0], logits[1]
    top_s, top_e = s.topk(5, dim=-1).indices, e.topk(5, dim=-1).indices
    null = s[:, 0] + e[:, 0]
    best = s[:, 1:].max(-1).values + e[:, 1:].max(-1).values
    return torch.cat([top_s, top_e, null[:, None], best[:, None]], dim=-1).float()


def compute_metrics(p) -> dict[str, float]:
    """Validation proxies; `pred_null_rate` near 1 means the model always answers 'absent'."""
    from sklearn.metrics import average_precision_score

    pred = np.asarray(p.predictions)
    top_s, top_e = pred[:, :5].astype(int), pred[:, 5:10].astype(int)
    null, best = pred[:, 10], pred[:, 11]
    gs, ge = (np.asarray(x) for x in p.label_ids)
    s1, e1 = top_s[:, 0], top_e[:, 0]
    pred_null, gold_null = (s1 == 0) & (e1 == 0), (gs == 0) & (ge == 0)
    pos = ~gold_null
    n = max(int(pos.sum()), 1)
    in_top5 = (top_s == gs[:, None]).any(1) & (top_e == ge[:, None]).any(1)
    return {
        "pred_null_rate": float(pred_null.mean()),
        "gold_null_rate": float(gold_null.mean()),
        "null_accuracy": float((pred_null == gold_null).mean()),
        "pos_exact": float(((s1 == gs) & (e1 == ge))[pos].sum() / n),
        "pos_top5": float(in_top5[pos].sum() / n),
        "answerability_ap": float(average_precision_score(pos.astype(int), best - null)),
    }


def hardware() -> tuple[bool, str, int, str]:
    """(bf16 available, attention implementation, world size, device name)."""
    import torch

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        return False, "sdpa", world, "cpu"
    major, minor = torch.cuda.get_device_capability(0)
    ampere = major * 10 + minor >= 80
    # fp16 with FlashAttention-2 gives NaN logits on ModernBERT models, so FA2 only with bf16.
    flash = ampere and importlib.util.find_spec("flash_attn") is not None
    return ampere, "flash_attention_2" if flash else "sdpa", world, torch.cuda.get_device_name(0)


def _is_main() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def log(*parts) -> None:
    if _is_main():
        print(*parts, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--features", required=True, help="directory with train/ and val/")
    parser.add_argument("--train-dir", help="override <features>/train")
    parser.add_argument("--val-dir", help="override <features>/val")
    parser.add_argument("--model", default="jhu-clsp/ettin-encoder-150m")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-epochs", type=float, default=3)
    parser.add_argument(
        "--final",
        choices=("last", "best"),
        default="last",
        help="last: the annealed last epoch; best: early stopping on --metric",
    )
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--metric", default="eval_loss")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=float, default=0.06, help="fraction of the total steps")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--effective-batch", type=int, default=16)
    parser.add_argument("--per-device-bs", type=int)
    parser.add_argument("--grad-acc", type=int, default=2)
    parser.add_argument("--eval-bs", type=int, default=8)
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--doc-stride", type=int, default=256)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--fp32", action="store_true")
    return parser.parse_args()


def _load(path: str, limit: int | None):
    from datasets import load_from_disk

    ds = load_from_disk(path).select_columns(list(FEATURE_COLUMNS)).with_format("torch")
    return ds.select(range(min(limit, len(ds)))) if limit else ds


def main() -> None:
    args = parse_args()

    import torch
    import transformers
    from transformers import (
        AutoModelForQuestionAnswering,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    from .windowing import assert_sliding_windows

    bf16, attn, world, device = hardware()
    if args.fp32 or device == "cpu":
        bf16, fp16 = False, False
    else:
        fp16 = not bf16
    per_device = args.per_device_bs or max(1, args.effective_batch // (world * args.grad_acc))
    effective = per_device * world * args.grad_acc
    if effective != args.effective_batch:
        raise SystemExit(f"effective batch {effective} != {args.effective_batch}")
    batch = f"{per_device}x{world}x{args.grad_acc}"
    log(f"{device} x {world} | bf16={bf16} fp16={fp16} attn={attn} | batch {batch}")

    train_ds = _load(args.train_dir or f"{args.features}/train", args.limit_train)
    val_ds = _load(args.val_dir or f"{args.features}/val", args.limit_val)
    log(f"train {len(train_ds)} | val {len(val_ds)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    assert_sliding_windows(tok, args.max_seq_len, args.doc_stride)
    # Trainer seeds only after the model exists, so the new QA head is seeded here.
    set_seed(args.seed)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model, attn_implementation=attn)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    out = Path(args.out)
    training_args = TrainingArguments(
        output_dir=str(out),
        seed=args.seed,
        learning_rate=args.lr,
        num_train_epochs=args.max_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=args.grad_acc,
        per_device_eval_batch_size=args.eval_bs,
        gradient_checkpointing=args.grad_ckpt,
        warmup_steps=args.warmup,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        fp16=fp16,
        bf16=bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=int(np.ceil(args.max_epochs)),
        save_only_model=True,
        load_best_model_at_end=args.final == "best",
        metric_for_best_model=args.metric,
        greater_is_better=not args.metric.endswith("loss"),
        logging_steps=args.logging_steps,
        report_to="none",
        label_names=["start_positions", "end_positions"],
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
    )
    callbacks = [EarlyStoppingCallback(args.patience)] if args.final == "best" else []
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
        processing_class=tok,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks,
    )

    start = time.time()
    train_out = trainer.train()
    final_eval = trainer.evaluate()
    if not trainer.is_world_process_zero():
        return

    state = trainer.state
    evals = [e for e in state.log_history if "eval_loss" in e]
    best_epoch = next((e["epoch"] for e in evals if e.get(args.metric) == state.best_metric), None)
    checkpoints = sorted(map(str, out.glob("checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    trainer.save_model(str(out / "final"))
    tok.save_pretrained(str(out / "final"))
    (out / "log_history.json").write_text(json.dumps(state.log_history, indent=1))
    summary = {
        "model": args.model,
        "features": args.features,
        "seed": args.seed,
        "effective_batch": effective,
        "per_device_bs": per_device,
        "world_size": world,
        "grad_acc": args.grad_acc,
        "lr": args.lr,
        "warmup": args.warmup,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "metric": args.metric,
        "epochs_run": state.epoch,
        "best_metric": state.best_metric,
        "best_epoch": best_epoch,
        "final": args.final,
        "final_epoch": best_epoch if args.final == "best" else state.epoch,
        "checkpoints": checkpoints,
        "steps_per_epoch": round(state.global_step / state.epoch) if state.epoch else None,
        "final_eval": final_eval,
        "train_metrics": train_out.metrics,
        "wall_s": time.time() - start,
        "device": device,
        "bf16": bf16,
        "fp16": fp16,
        "attn": attn,
        "grad_ckpt": args.grad_ckpt,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    minutes = summary["wall_s"] / 60
    log(f"{state.epoch:g} epochs in {minutes:.0f} min; checkpoints: {len(checkpoints)}")
    for key in sorted(final_eval):
        log(f"  {key:26s} {final_eval[key]:.4f}")


if __name__ == "__main__":
    main()
