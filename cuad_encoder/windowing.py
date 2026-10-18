"""Check that the tokenizer still produces sliding windows over long inputs."""

from __future__ import annotations

_SENTENCE = "This Agreement shall be governed by the laws of the State of New York. "


class SlidingWindowError(RuntimeError):
    pass


def assert_sliding_windows(tok, max_seq_len: int, doc_stride: int) -> int:
    """Raise unless overflow windowing covers a long synthetic text; return the window count.

    transformers 5.16 and 5.17 silently return only two windows for a long input, which
    cuts every contract to its first pages.
    """
    text = (_SENTENCE * 12 + "\n") * max(120, (3 * max_seq_len) // 150)
    enc = tok(
        "Which law governs?",
        text,
        truncation="only_second",
        max_length=max_seq_len,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    n = len(enc["input_ids"])
    seq_ids = enc.sequence_ids(n - 1)
    ctx = [k for k, s in enumerate(seq_ids) if s == 1]
    covered = enc["offset_mapping"][n - 1][ctx[-1]][1] / len(text) if ctx else 0.0

    if n < 3 or covered < 0.98:
        import transformers

        raise SlidingWindowError(
            f"tokenizer produced {n} window(s) covering {covered:.0%} of the text "
            f"(transformers {transformers.__version__}); install transformers>=5.13,<5.16"
        )
    return n
