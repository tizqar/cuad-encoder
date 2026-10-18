import math

from cuad_encoder.qa import nbest

CONTEXT = "alpha beta gamma delta"


def test_duplicates_are_merged_before_the_softmax():
    # Two windows produce the same span; it must count once, at its higher score.
    prelim = [(2.0, 0, 5), (1.0, 0, 5), (1.5, 6, 10)]
    ranked = nbest(prelim, null_score=0.0, context=CONTEXT, n_best_size=20, top_k=20)
    assert [t for t, _ in ranked] == ["alpha", "beta"]
    total = math.exp(2.0) + math.exp(1.5) + math.exp(0.0)
    assert abs(ranked[0][1] - math.exp(2.0) / total) < 1e-12


def test_null_score_takes_probability_mass():
    confident = nbest([(5.0, 0, 5)], null_score=0.0, context=CONTEXT, n_best_size=20, top_k=1)
    hesitant = nbest([(5.0, 0, 5)], null_score=10.0, context=CONTEXT, n_best_size=20, top_k=1)
    assert confident[0][1] > 0.99
    assert hesitant[0][1] < 0.01


def test_n_best_size_and_top_k_limit_the_candidates():
    prelim = [(3.0, 0, 5), (2.0, 6, 10), (1.0, 11, 16)]
    assert len(nbest(prelim, 0.0, CONTEXT, n_best_size=2, top_k=20)) == 2
    assert len(nbest(prelim, 0.0, CONTEXT, n_best_size=20, top_k=1)) == 1


def test_blank_spans_and_empty_input_give_no_candidates():
    assert nbest([], 0.0, CONTEXT, 20, 20) == []
    assert nbest([(1.0, 5, 6)], 0.0, CONTEXT, 20, 20) == []
