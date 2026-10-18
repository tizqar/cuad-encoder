# cuad-encoder

Fine-tuning Ettin encoders ([jhu-clsp/ettin-encoder-150m](https://huggingface.co/jhu-clsp/ettin-encoder-150m)
and [-400m](https://huggingface.co/jhu-clsp/ettin-encoder-400m)) for clause extraction on
[CUAD](https://github.com/TheAtticusProject/cuad), framed as extractive question answering
over whole contracts with sliding windows, and evaluated with CUAD's metric on the
official test split.

## Results

CUAD's official test split (102 contracts), CUAD's threshold grid.

| model | params | AUPR | P@80R | P@90R |
|---|---|---|---|---|
| Ettin-400M, seed 3407 | 400M | **57.31** | **53.86** | 0.00 |
| Ettin-150M, seeds 3407 / 42 / 1234 (mean ± sd) | 150M | **52.78 ± 0.73** | **40.54 ± 0.21** | 0.00 |
| RoBERTa-base CUAD checkpoint, this evaluator | 125M | 39.87 | 13.69 | 0.00 |
| RoBERTa-base, published | 125M | 42.6 | 31.1 | 0.0 |
| RoBERTa-large, published | 355M | 48.2 | 38.1 | 0.0 |
| DeBERTa-v2-xlarge, published | 900M | 47.8 | 44.0 | 17.8 |

Published rows: Hendrycks et al. 2021, Table 2. The RoBERTa-base row is
`akdeniz27/roberta-base-cuad` run through `cuad_encoder.evaluate` with 384/128 windows;
only six of its per-type rows were kept (`results/roberta-base-cuad_partial_by_type.csv`).

Epochs 2 and 3 are scored on validation and every run selected epoch 3. Training took 80 minutes for the
150M model and 193 minutes for the 400M model on Kaggle's T4 x2.

| run | val AUPR, epoch 2 | val AUPR, epoch 3 | test AUPR |
|---|---|---|---|
| 150M seed 3407 | 51.78 | 52.52 | 53.21 |
| 150M seed 42 | 45.57 | 52.17 | 51.94 |
| 150M seed 1234 | 50.72 | 52.07 | 53.21 |
| 400M seed 3407 | 50.72 | 58.03 | 57.31 |

## Protocol

- **Splits.** CUAD's 102 test contracts; 42 of the 408 training contracts held out for
  validation by a hash of the title (`data/splits.json`).
- **Features.** 1024-token windows overlapping by 256 tokens. A window holding several
  gold spans of a clause type yields one positive per span (`--answer-mode multi`).
  Negatives are subsampled to one per positive for training, three for validation.
  Every positive's token span is checked to reconstruct its gold text.
- **Training.** Full fine-tuning, AdamW, learning rate 5e-5, 6% warmup, weight decay
  0.01, effective batch 16, linear schedule over 3 epochs, a checkpoint per epoch. fp16
  with SDPA attention on T4; DDP over both GPUs. The seed fixes the data order and the
  initialisation of the QA head. In the runs reported above the head was initialised
  before seeding, so their spread over seeds also includes the head initialisation.
- **Epoch selection.** Each checkpoint is scored with the CUAD metric on the validation
  contracts and the best one becomes the model; the test split is scored once.
- **Inference.** Windows and n-best postprocessing as in CUAD's `utils_qa.py`: a span may
  start only in the window where its first token is most central, the 20 best unique
  spans are softmaxed against the weakest no-answer score.
- **Thresholds.** `cuad_encoder.thresholds` picks two confidence levels on validation:
  `tau_low` maximises citation F1 (0.06), `tau_high` is the lowest with citation precision
  of at least 0.75 (0.91).

## Layout

| module | role |
|---|---|
| `cuad_encoder/data.py` | download CUAD, write `data/splits.json` and `data/raw/val.json` |
| `cuad_encoder/features.py` | windowed, labelled, balanced training features |
| `cuad_encoder/train.py` | Trainer script, one checkpoint per epoch |
| `cuad_encoder/select_epoch.py` | validation AUPR per checkpoint, installs `final/` |
| `cuad_encoder/evaluate.py` | inference over whole contracts and CUAD scoring |
| `cuad_encoder/metrics.py` | CUAD's metric (Jaccard >= 0.5, AUPR, precision at recall) |
| `cuad_encoder/pipeline.py` | train, select and evaluate for each seed |
| `cuad_encoder/thresholds.py` | operating thresholds from validation predictions |
| `cuad_encoder/figures.py` | seed means, comparison figures and tables |

`results/` holds the reports of the runs above (`<run>.json`, `<run>_by_type.csv`, the
`_dense` variants, validation reports per epoch), the run records in `results/runs/`,
the seed summaries, the operating point and the figures. Prediction files are not kept
in git.

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
python -m cuad_encoder.data
python -m cuad_encoder.features --tokenizer jhu-clsp/ettin-encoder-150m
python -m cuad_encoder.pipeline --features data/features/ettin-encoder-150m_w1024_s256_multi_neg1 \
    --seeds 3407 42 1234 --gpus 2
```

For the 400M model build the features with `--tokenizer jhu-clsp/ettin-encoder-400m` and
run the pipeline on `data/features/ettin-encoder-400m_w1024_s256_multi_neg1` with
`--model jhu-clsp/ettin-encoder-400m --grad-ckpt`. On Kaggle, run
`notebooks/kaggle_train.ipynb` with the archive from `make kaggle-zip` attached.

Scoring saved predictions again, and the seed mean:

```bash
python -m cuad_encoder.evaluate --tag <run> --score-only
python -m cuad_encoder.evaluate --tag <run> --score-only --grid dense --curve
python -m cuad_encoder.figures mean --tag <prefix>_e3v_SEEDMEAN results/<prefix>_seed*_e3v.json
```

Training is meant for CUDA GPUs; on Apple's MPS backend a short test run produced a
meaningless loss, while the same run on CPU trained normally.

transformers is pinned below 5.16: from 5.16 the tokenizer returns only the first windows
of a long contract. `cuad_encoder.windowing` checks for this before features are built and
before evaluation.

## Tests

```bash
make test
make lint
```
