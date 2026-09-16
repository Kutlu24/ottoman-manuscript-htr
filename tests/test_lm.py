from ottoman_htr.language.lm import CharNGramLanguageModel


def test_lm_prefers_seen_pattern():
    lm = CharNGramLanguageModel(order=3)
    lm.fit(["devlet-i aliyye"] * 20)

    best, _ = lm.rescore_candidates(["devlet-i aliyye", "xzqjw plmnv"])[0]

    assert best == "devlet-i aliyye"
