import torch

from ottoman_htr.data.vocab import CharVocab
from ottoman_htr.metrics import (
    character_error_rate,
    corpus_cer,
    corpus_wer,
    greedy_ctc_decode,
    word_error_rate,
)


def test_character_error_rate_exact_match():
    assert character_error_rate("ابت", "ابت") == 0.0


def test_character_error_rate_one_substitution():
    assert character_error_rate("ابت", "ابث") == 1 / 3


def test_character_error_rate_empty_target():
    assert character_error_rate("", "") == 0.0
    assert character_error_rate("x", "") == 1.0


def test_word_error_rate_counts_word_edits():
    assert word_error_rate("bir iki uc", "bir iki dort") == 1 / 3


def test_corpus_cer_is_micro_averaged_not_mean_of_means():
    # a 1-char target with 1 error should not dominate a 10-char target with 1 error
    preds = ["x", "abcdefghiy"]
    targets = ["y", "abcdefghij"]
    # total edits = 1 + 1 = 2, total chars = 1 + 10 = 11
    assert corpus_cer(preds, targets) == 2 / 11


def test_corpus_wer_matches_manual_computation():
    preds = ["bir iki"]
    targets = ["bir uc"]
    assert corpus_wer(preds, targets) == 1 / 2


def test_greedy_ctc_decode_collapses_repeats_and_drops_blank():
    vocab = CharVocab(["a", "b"])
    # itos = [<blank>, <unk>, a, b] -> indices 0,1,2,3
    # sequence: a a <blank> b b b -> collapse repeats -> a <blank> b -> drop blank -> "ab"
    seq = [2, 2, 0, 3, 3, 3]
    logits = torch.full((1, len(seq), len(vocab)), -10.0)
    for t, idx in enumerate(seq):
        logits[0, t, idx] = 10.0
    log_probs = torch.log_softmax(logits, dim=-1)

    decoded = greedy_ctc_decode(log_probs, vocab)
    assert decoded == ["ab"]
