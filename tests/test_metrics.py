import numpy as np

from cuad_encoder.metrics import (
    EXCLUDED_CLAUSE_TYPES,
    IOU_THRESH,
    compute_precision_recall,
    evaluate,
    get_aupr,
    get_jaccard,
    get_prec_at_recall,
    get_precisions_recalls,
    is_match,
    process_precisions,
    threshold_preds,
)


def test_jaccard_normalises_punctuation_and_case():
    # {governed, by, the, laws} against six words: 4 / 6
    assert (
        abs(get_jaccard("Governed by, the laws.", "governed by the laws of California") - 4 / 6)
        < 1e-9
    )
    assert get_jaccard("a b c", "a b c") == 1.0
    assert get_jaccard("net 30/60", "net 30 60") == 1.0


def test_jaccard_of_one_half_is_a_match():
    assert get_jaccard("a b", "a b c d") == 0.5
    assert is_match("a b", "a b c d") is True
    assert get_jaccard("a b", "a b c d e") == 0.4
    assert is_match("a b", "a b c d e") is False
    assert IOU_THRESH == 0.5


def test_parties_match_by_substring_only_when_relaxed():
    gold = "Acme Corp"
    pred = "this agreement is between Acme Corp and Beta LLC as of today ok"
    assert get_jaccard(gold, pred) < IOU_THRESH
    assert is_match(gold, pred, relaxed=False) is False
    assert is_match(gold, pred, relaxed=True) is True


def test_true_positives_count_golds_and_false_positives_count_predictions():
    # G1 found, G2 missed; the duplicate of P1 is not a false positive, P3 is.
    gt = {("c1", "Governing Law"): ["alpha beta gamma", "delta epsilon zeta"]}
    preds = {("c1", "Governing Law"): ["alpha beta gamma", "alpha beta gamma", "nothing like it"]}
    assert compute_precision_recall(gt, preds) == (0.5, 0.5)


def test_predictions_for_an_absent_clause_are_false_positives():
    gt = {("c1", "Governing Law"): ["alpha beta gamma"], ("c1", "Non-Compete"): []}
    preds = {
        ("c1", "Governing Law"): ["alpha beta gamma"],
        ("c1", "Non-Compete"): ["spurious one", "spurious two"],
    }
    precision, recall = compute_precision_recall(gt, preds)
    assert abs(precision - 1 / 3) < 1e-9
    assert recall == 1.0


def test_undefined_precision_and_recall_are_nan():
    precision, recall = compute_precision_recall(
        {("c1", "Governing Law"): ["alpha beta gamma"]}, {("c1", "Governing Law"): []}
    )
    assert np.isnan(precision) and recall == 0.0
    precision, recall = compute_precision_recall(
        {("c1", "Price Restrictions"): []}, {("c1", "Price Restrictions"): []}
    )
    assert np.isnan(precision) and np.isnan(recall)


def test_threshold_preds_filters_by_confidence_and_drops_blank_spans():
    preds = {("c1", "X"): [("keep", 0.9), ("drop", 0.4), ("   ", 0.99)]}
    assert threshold_preds(preds, 0.5) == {("c1", "X"): ["keep"]}
    assert threshold_preds(preds, 0.95) == {("c1", "X"): []}


def test_precision_envelope_is_a_running_max_from_the_right():
    assert process_precisions([1, 0.2, 0.9, 0.3]) == [1, 0.9, 0.9, 0.3]
    assert process_precisions([1.0, float("nan"), 0.5]) == [1.0, 0.5, 0.5]


def test_aupr_of_two_trapezoids():
    # 0.5 * (1 + 1) / 2 + 0.5 * (1 + 0.5) / 2
    assert abs(get_aupr([1, 1, 0.5], [0, 0.5, 1.0]) - 0.875) < 1e-9


def test_aupr_is_zero_when_recall_is_undefined():
    assert get_aupr([1, float("nan")], [0, float("nan")]) == 0.0


def test_precision_at_recall_uses_the_first_point_reaching_it():
    p, r = [1, 0.9, 0.7, 0.6], [0, 0.5, 0.85, 0.95]
    assert get_prec_at_recall(p, r, 0.8) == 0.7
    assert get_prec_at_recall(p, r, 0.9) == 0.6
    assert get_prec_at_recall(p, r, 0.99) == 0.0


def test_perfect_predictions_score_one():
    gt = {
        ("c1", "Governing Law"): ["governed by the laws of delaware"],
        ("c2", "Governing Law"): ["governed by the laws of new york"],
    }
    out = evaluate(gt, {k: [(v[0], 1.0)] for k, v in gt.items()})
    assert abs(out["overall"]["aupr"] - 1.0) < 1e-9
    assert out["overall"]["prec_at_80_recall"] == 1.0
    assert out["overall"]["prec_at_90_recall"] == 1.0


def test_price_restrictions_count_overall_but_not_in_the_macro_average():
    gt = {
        ("c1", "Governing Law"): ["governed by the laws of delaware"],
        ("c1", "Price Restrictions"): [],
    }
    preds = {
        ("c1", "Governing Law"): [("governed by the laws of delaware", 1.0)],
        ("c1", "Price Restrictions"): [("some spurious clause text", 1.0)],
    }
    out = evaluate(gt, preds)
    assert out["excluded_clause_types"] == ["Price Restrictions"]
    assert out["macro"]["macro_aupr"] == 1.0
    assert out["overall"]["aupr"] < 1.0
    assert EXCLUDED_CLAUSE_TYPES == ("Price Restrictions",)


def test_confident_correct_answers_give_a_better_curve():
    gt = {
        ("c1", "Governing Law"): ["alpha beta gamma delta"],
        ("c2", "Governing Law"): ["epsilon zeta eta theta"],
    }
    good = {
        ("c1", "Governing Law"): [("alpha beta gamma delta", 0.9)],
        ("c2", "Governing Law"): [("totally unrelated text here", 0.2)],
    }
    bad = {
        ("c1", "Governing Law"): [("alpha beta gamma delta", 0.2)],
        ("c2", "Governing Law"): [("totally unrelated text here", 0.9)],
    }
    good_p, good_r, _ = get_precisions_recalls(gt, good)
    bad_p, bad_r, _ = get_precisions_recalls(gt, bad)
    assert get_aupr(good_p, good_r) > get_aupr(bad_p, bad_r)
