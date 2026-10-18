import json

import pytest

from cuad_encoder.evaluate import gold_dict, load_eval_set, write_results
from cuad_encoder.figures import draw, load_by_type, mean_report, sibling_share
from cuad_encoder.metrics import evaluate


def report(cuad_json, confidence: float) -> dict:
    gt = gold_dict(load_eval_set(cuad_json))
    preds = {k: [(spans[0], confidence)] if spans else [] for k, spans in gt.items()}
    out = evaluate(gt, preds)
    out["config"] = {"model": f"model-{confidence}"}
    return out


def test_mean_report_round_trips_through_the_by_type_csv(cuad_json, tmp_path):
    reports = [report(cuad_json, 0.9), report(cuad_json, 0.8)]
    write_results(mean_report(reports), tmp_path, "mean")
    loaded = load_by_type(tmp_path / "mean_by_type.csv")
    expected = sum(r["overall"]["aupr"] for r in reports) / 2 * 100
    assert loaded["overall"]["aupr"] == pytest.approx(expected, abs=0.01)
    assert set(loaded["types"]) == {"Governing Law", "Parties", "Anti-Assignment"}


def test_sibling_share_counts_second_spans(cuad_json):
    assert sibling_share(cuad_json)["Parties"] == (4, 0.5)
    assert sibling_share(cuad_json)["Governing Law"] == (2, 0.0)


def test_every_figure_renders(cuad_json, tmp_path):
    pytest.importorskip("matplotlib")
    write_results(report(cuad_json, 0.9), tmp_path, "a")
    write_results(report(cuad_json, 0.5), tmp_path, "b")
    preds = tmp_path / "a_preds.jsonl"
    gt = gold_dict(load_eval_set(cuad_json))
    with preds.open("w") as fh:
        for cid in ("contract-a", "contract-b"):
            record = {
                ct: [[spans[0], 0.9]] if spans else [] for (c, ct), spans in gt.items() if c == cid
            }
            fh.write(json.dumps({"contract_id": cid, "preds": record}) + "\n")
    history = tmp_path / "log_history.json"
    history.write_text(
        json.dumps(
            [{"loss": 1.0, "epoch": 0.5}, {"eval_loss": 0.7, "eval_pos_exact": 0.4, "epoch": 1.0}]
        )
    )

    systems = {"a": tmp_path / "a_by_type.csv", "b": tmp_path / "b_by_type.csv"}
    made = draw(tmp_path / "figures", systems, {"a": preds}, cuad_json, history)
    assert {p.name for p in made} == {
        "summary.csv",
        "per_type.csv",
        "run_comparison.png",
        "per_type_aupr.png",
        "sibling_share_vs_aupr.png",
        "pr_curves.png",
        "training_curves.png",
    }
    assert all(p.stat().st_size > 0 for p in made)
