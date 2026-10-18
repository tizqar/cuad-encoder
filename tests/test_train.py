from types import SimpleNamespace

import numpy as np
import torch

from cuad_encoder.train import collate, compute_metrics, preprocess_logits_for_metrics


def test_logits_are_reduced_to_top_spans_and_scores():
    start = torch.tensor([[5.0, 1.0, 9.0, 0.0, 0.0, 0.0]])
    end = torch.tensor([[5.0, 0.0, 0.0, 8.0, 0.0, 0.0]])
    out = preprocess_logits_for_metrics((start, end), None)
    assert out.shape == (1, 12)
    assert out[0, 0] == 2 and out[0, 5] == 3
    assert out[0, 10] == 10.0 and out[0, 11] == 17.0


def test_compute_metrics_on_a_null_and_a_positive():
    predictions = np.zeros((2, 12))
    predictions[1, :5] = [4, 1, 2, 3, 5]
    predictions[1, 5:10] = [6, 1, 2, 3, 4]
    predictions[:, 10] = [5.0, 1.0]
    predictions[:, 11] = [1.0, 5.0]
    labels = (np.array([0, 4]), np.array([0, 6]))
    m = compute_metrics(SimpleNamespace(predictions=predictions, label_ids=labels))
    assert m["null_accuracy"] == 1.0
    assert m["pos_exact"] == 1.0
    assert m["answerability_ap"] == 1.0


def test_collate_stacks_the_model_inputs():
    item = {k: torch.tensor([1, 2]) for k in ("input_ids", "attention_mask")}
    item |= {"start_positions": torch.tensor(0), "end_positions": torch.tensor(1)}
    batch = collate([item, item])
    assert batch["input_ids"].shape == (2, 2)
    assert batch["start_positions"].dtype == torch.long
