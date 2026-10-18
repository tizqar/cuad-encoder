import math

from cuad_encoder.thresholds import choose_tau_high, choose_tau_low, outcome, score

GT = {
    ("c", "Governing Law"): ["governed by the laws of delaware"],
    ("c", "Non-Compete"): [],
    ("c", "Audit Rights"): ["the auditor may inspect the books"],
}
TEXTS = {"c": "governed by the laws of delaware. the auditor may inspect the books."}


def test_outcomes_of_the_top_candidate():
    assert outcome(["a b"], "a b", 0.9, 0.5, relaxed=False) == "correct"
    assert outcome(["a b"], "x y z w", 0.9, 0.5, relaxed=False) == "wrong"
    assert outcome(["a b"], "a b", 0.1, 0.5, relaxed=False) == "omission"
    assert outcome([], "a b", 0.9, 0.5, relaxed=False) == "false"
    assert outcome([], "a b", 0.1, 0.5, relaxed=False) == "abstain"
    assert outcome(["a b"], "", 0.9, 0.5, relaxed=False) == "omission"


def test_score_counts_and_rates():
    preds = {
        ("c", "Governing Law"): [("governed by the laws of delaware", 0.9)],
        ("c", "Non-Compete"): [("governed by the laws", 0.6)],
        ("c", "Audit Rights"): [("the auditor may inspect the books", 0.3)],
    }
    r = score(GT, preds, TEXTS, tau=0.5)
    assert (r["correct"], r["false"], r["omission"]) == (1, 1, 1)
    assert r["citation_precision"] == 0.5
    assert r["citation_f1"] == 2 / 4
    assert r["false_citation_rate"] == 1.0
    assert r["verbatim_rate"] == 1.0
    assert math.isnan(score(GT, {}, TEXTS, tau=0.5)["citation_precision"])


def test_thresholds_trade_precision_for_recall():
    preds = {
        ("c", "Governing Law"): [("governed by the laws of delaware", 0.9)],
        ("c", "Non-Compete"): [("governed by the laws", 0.6)],
        ("c", "Audit Rights"): [("the auditor may inspect the books", 0.3)],
    }
    tau_low, sweep = choose_tau_low(GT, preds, TEXTS)
    assert tau_low == 0.3
    assert choose_tau_high(sweep, target=1.0) == 0.61
    assert choose_tau_high(sweep, target=1.1) is None
