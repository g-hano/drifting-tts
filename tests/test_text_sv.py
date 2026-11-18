"""Swedish text frontend (``language: sv``): normalisation, the IPA vocabulary and the language switch."""

import pytest

from drifting_tts import text_sv
from drifting_tts.text import BLANK_ID, PAD_ID, frontend

CASES = [
    ("Hej, jag heter Anna.", "hej, jag heter anna."),
    ("Klockan är 12:30 och tåget går 07:05.", "klockan är tolv trettio och tåget går sju noll fem."),
    (
        "Det kostar 500 kr, 1 kr och 1 250 kr.",
        "det kostar femhundra kronor, en krona och ettusen tvåhundrafemtio kronor.",
    ),
    ("Det är 20% rabatt och 20°C ute.", "det är tjugo procent rabatt och tjugo grader celsius ute."),
    (
        "Jag kommer den 5 januari 2024, 2024-01-05 eller 5/1/2024.",
        "jag kommer den femte januari tvåtusentjugofyra, femte januari tvåtusentjugofyra eller femte januari "
        "tvåtusentjugofyra.",
    ),
    ("Vi ses den 1 maj och 2024/05/01.", "vi ses den första maj och första maj tvåtusentjugofyra."),
    (
        "Det väger 5 kg, är 10 km bort och går i 100 km/h.",
        "det väger fem kilogram, är tio kilometer bort och går i hundra kilometer i timmen.",
    ),
    ("Vi ses kl. 15:00, t.ex. på St. Eriksplan.", "vi ses klockan femton, till exempel på sankt eriksplan."),
    (
        "Se www.example.se/test-sida eller mejla john.doe@example.com.",
        "se ve ve ve punkt example punkt se snedstreck test streck sida eller mejla john punkt doe snabel-a example "
        "punkt com.",
    ),
    ("AB, USA, EU och PDF.", "aktiebolag, u ess a, e u och pe de eff."),
    ("Läs 1/2, 3/4 och 1 1/2 timme.", "läs en halv, tre fjärdedelar och en och en halv timme."),
    ("nr 7, 21:a och 3:e.", "nummer sju, tjugoförsta och tredje."),
    ("Det är -5 grader, 1999 och 2024.", "det är minus fem grader, nittonhundranittionio och tvåtusentjugofyra."),
    ("Ring 070-123 45 67.", "ring noll sju noll ett två tre fyra fem sex sju."),
    ("Carl XVI Gustaf och Elizabeth II.", "carl sextonde gustaf och elizabeth andra."),
    ("t.ex. jag vet inte. T.ex. Jag vet.", "till exempel jag vet inte. till exempel. jag vet."),
]


@pytest.mark.parametrize("raw,expected", CASES)
def test_normalize(raw, expected):
    assert text_sv.normalize(raw) == expected


@pytest.mark.parametrize("raw", [raw for raw, _ in CASES])
def test_normalize_is_idempotent(raw):
    norm = text_sv.normalize(raw)
    assert text_sv.normalize(norm) == norm


@pytest.mark.parametrize(
    "raw, sentences",
    [
        ("Hej! Hur mår du? Jag mår bra.", ["hej!", "hur mår du?", "jag mår bra."]),
        ("Vi ses kl. 15:00. T.ex. på St. Eriksplan.", ["vi ses klockan femton.", "till exempel på sankt eriksplan."]),
        (
            "Det kostar 500 kr. Ring 070-123 45 67. Tack.",
            ["det kostar femhundra kronor.", "ring noll sju noll ett två tre fyra fem sex sju.", "tack."],
        ),
    ],
)
def test_split_sentences(raw, sentences):
    assert text_sv.split_sentences(text_sv.normalize(raw)) == sentences


def test_numbers():
    assert text_sv.number_to_words(0) == "noll"
    assert text_sv.number_to_words(1250) == "ettusen tvåhundrafemtio"
    assert text_sv.number_to_words(-12) == "minus tolv"
    assert text_sv.number_to_words(2_500_000) == "två miljoner femhundratusen"
    assert text_sv.ordinal_to_words(22) == "tjugoandra"
    assert text_sv.ordinal_to_words(12) == "tolfte"
    assert text_sv.year_to_words(2015) == "tvåtusenfemton"
    assert text_sv.year_to_words(1905) == "nittonhundrafem"


def test_symbols_and_frontend_switch():
    sv, tr = frontend("sv"), frontend("tr")
    assert sv is text_sv and tr.SYMBOLS[:2] == sv.SYMBOLS[:2]
    assert (sv.PAD_ID, sv.BLANK_ID) == (PAD_ID, BLANK_ID)
    # IPA, not orthography: no Swedish letters, but stress, length and the sj/tj sounds are symbols
    assert "å" not in sv.SYMBOLS and "ä" not in sv.SYMBOLS and "ö" not in sv.SYMBOLS
    assert {"ˈ", "ˌ", "ː", "ɕ", "ɧ", "ʉ", "ø"} <= set(sv.SYMBOLS)
    assert " " in sv.SYMBOLS and all(c in sv.SYMBOL_TO_ID for c in ".,!?")
    with pytest.raises(ValueError):
        frontend("de")


def test_phonemized_ids_round_trip():
    ipa = "hˈɛj, jɑːɡ hˈeːtər ˈana."
    ids = text_sv.text_to_ids(ipa, phonemized=True)
    assert ids[0::2] == [BLANK_ID] * (len(ids) // 2 + 1)
    assert text_sv.ids_to_text(ids) == ipa
    assert text_sv.text_to_ids(ipa, phonemized=True, intersperse_blank=False) == [text_sv.SYMBOL_TO_ID[c] for c in ipa]


def _phonemize_or_skip(text: str) -> str:
    pytest.importorskip("phonemizer")
    try:
        return text_sv.phonemize(text)
    except (ImportError, RuntimeError) as e:  # no espeak-ng on this machine
        pytest.skip(str(e))


def test_phonemize_swedish():
    assert _phonemize_or_skip("Hej!").startswith("h")
    out = _phonemize_or_skip("Hej, jag heter Anna.")
    assert "ˈ" in out and out.endswith(".")
    assert set(out) <= set(text_sv.SYMBOL_TO_ID)
    assert "ɧ" in _phonemize_or_skip("sjukhus")  # the sj sound gets one symbol


def test_text_to_ids_phonemizes_normalised_text():
    pytest.importorskip("phonemizer")
    try:
        ids = text_sv.text_to_ids("tolv trettio", normalized=True)
    except (ImportError, RuntimeError) as e:
        pytest.skip(str(e))
    expected = text_sv.text_to_ids(text_sv.phonemize("tolv trettio", normalized=True), phonemized=True)
    assert ids == expected


def test_swedish_syllables():
    pytest.importorskip("torch")
    from drifting_tts.prosody import syllables

    assert syllables("hej jag heter anna", "sv") == 6
    assert syllables("sjukhus", "sv") == 2
    assert syllables("", "sv") == 0


def test_swedish_ar_model_vocabulary():
    pytest.importorskip("torch")
    from drifting_tts.config import load_config
    from drifting_tts.models.ar import ARDriftTTS

    cfg = load_config("configs/ar_tts.yaml", ["model.text.language=sv", "model.ar.depth=1", "model.ar.head_depth=1"])
    model = ARDriftTTS(cfg.model, num_speakers=2)
    assert model.encoder.emb.num_embeddings == len(text_sv.SYMBOLS)
