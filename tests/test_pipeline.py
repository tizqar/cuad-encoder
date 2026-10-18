import json

from cuad_encoder.pipeline import Config, summarize


def test_run_name_comes_from_the_feature_directory():
    cfg = Config(features="data/features/ettin-encoder-150m_w1024_s256_multi_neg1")
    assert cfg.run_name(42) == "ettin-encoder-150m_w1024_s256_multi_neg1_seed42_e3v"


def test_summary_reports_mean_and_sample_deviation(tmp_path):
    rows = [
        {
            "seed": s,
            "tag": "t",
            "selected_epoch": 3.0,
            "val_aupr": 50.0,
            "aupr": a,
            "p80": 40.0,
            "p90": 0.0,
            "macro_aupr": 50.0,
        }
        for s, a in ((1, 51.94), (2, 53.21), (3, 53.21))
    ]
    aggregate = summarize(rows, tmp_path / "summary.csv")
    assert aggregate["aupr"]["mean"] == 52.79
    assert aggregate["aupr"]["std"] == 0.73
    assert json.loads((tmp_path / "summary.json").read_text())["seeds"] == rows
