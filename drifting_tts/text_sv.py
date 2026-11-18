"""Swedish text frontend: normalisation, espeak-ng phonemisation and an IPA vocabulary (``language: sv``).

Swedish orthography is not phonemic enough for a character vocabulary to be the best choice: ``k`` is [k] or [ɕ],
``g`` is [ɡ], [j] or [ɧ]-coloured, and the sj/tj sounds have several spellings.  The frontend therefore normalises
raw text (numbers, clock times, money, units, dates, abbreviations, acronyms, addresses) to ordinary lower-case
Swedish, then converts that to IPA with `phonemizer <https://github.com/bootphon/phonemizer>`_ and espeak-ng's
built-in ``sv`` voice.  The model's symbols are phonemes, punctuation and word spaces, not letters.

The phonemizer dependency is optional (``pip install phonemizer``) but espeak-ng must be installed on the system
(``brew install espeak-ng``, ``apt install espeak-ng``).  Phonemisation is lazy: importing this module and calling
:func:`normalize` work without either, which keeps data inspection and the other frontends usable.

Swedish has two lexical pitch accents (accent 1 and accent 2).  espeak-ng does not expose them in its IPA output:
``anden`` (the duck) and ``anden`` (the spirit) have the same phonemes.  The acoustic model can learn the accent
distribution from a sufficiently large corpus; a lexicon-driven accent tagger can be added on top of
:func:`phonemize` later without changing the normaliser.  See ``docs/SWEDISH.md``.
"""

from __future__ import annotations

import re
import unicodedata

from .text import BLANK, PAD, PUNCTUATION, group_sentences, split_sentences  # noqa: F401 (re-exported)

LANGUAGE = "sv"
ESPEAK_VOICE = "sv"
# Phonemes are more numerous than letters (stress and length marks), so the synthesiser's character budgets for
# sentences / context (180 / 190 characters through the 384-token CUDA-graph bucket) are smaller here.
PARAGRAPH_CHARS = 150
CONTEXT_CHARS = 160

# IPA symbols espeak-ng's Swedish voice can emit.  It also emits the phoneme name ``sx`` for /ɧ/; :func:`phonemize`
# rewrites that to one symbol.  The extra retroflex/palatal symbols are not all used by the current voice, but keeping
# them in the table makes a future espeak-ng or dictionary change a non-event.
PHONEMES = "".join(
    dict.fromkeys(
        "abcdefghijklmnoprstuvwxyz"  # IPA letters that are written like Latin letters
        "ðøŋœɐɑɒɔɕəɛɜɡɪɵʉʊʌʏɤɨɯɞɶ"  # Swedish vowels and the consonants written in IPA
        "ɧçʈɖɳɭɽɻɾɱɲʎɫʔɦʋʐʑʝɣ"  # sj/tj, retroflexes and marginal phones
        "ˈˌːˑʰʲʷ"  # stress, length and secondary articulations
    )
)
LETTERS = "abcdefghijklmnopqrstuvwxyzåäöé"
SYMBOLS = [PAD, BLANK, *PHONEMES, *PUNCTUATION]
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
PAD_ID, BLANK_ID = SYMBOL_TO_ID[PAD], SYMBOL_TO_ID[BLANK]

_ONES = ["noll", "ett", "två", "tre", "fyra", "fem", "sex", "sju", "åtta", "nio"]
_TEENS = {
    10: "tio",
    11: "elva",
    12: "tolv",
    13: "tretton",
    14: "fjorton",
    15: "femton",
    16: "sexton",
    17: "sjutton",
    18: "arton",
    19: "nitton",
}
_TENS = ["", "", "tjugo", "trettio", "fyrtio", "femtio", "sextio", "sjuttio", "åttio", "nittio"]
_SCALES = [(10**12, "biljon", "biljoner"), (10**9, "miljard", "miljarder"), (10**6, "miljon", "miljoner")]
_ORDINALS = {
    0: "nollte",
    1: "första",
    2: "andra",
    3: "tredje",
    4: "fjärde",
    5: "femte",
    6: "sjätte",
    7: "sjunde",
    8: "åttonde",
    9: "nionde",
    10: "tionde",
    11: "elfte",
    12: "tolfte",
    13: "trettonde",
    14: "fjortonde",
    15: "femtonde",
    16: "sextonde",
    17: "sjuttonde",
    18: "artonde",
    19: "nittonde",
    20: "tjugonde",
    30: "trettionde",
    40: "fyrtionde",
    50: "femtionde",
    60: "sextionde",
    70: "sjuttionde",
    80: "åttionde",
    90: "nittionde",
}
_MONTHS = [
    "januari",
    "februari",
    "mars",
    "april",
    "maj",
    "juni",
    "juli",
    "augusti",
    "september",
    "oktober",
    "november",
    "december",
]
_MONTH_ABBREVIATIONS = {
    "jan": "januari",
    "feb": "februari",
    "mar": "mars",
    "apr": "april",
    "jun": "juni",
    "jul": "juli",
    "aug": "augusti",
    "sep": "september",
    "sept": "september",
    "okt": "oktober",
    "nov": "november",
    "dec": "december",
}
_WEEKDAYS = {
    "mån": "måndag",
    "tis": "tisdag",
    "ons": "onsdag",
    "tors": "torsdag",
    "fre": "fredag",
    "lör": "lördag",
    "sön": "söndag",
}
_LETTER_NAMES = dict(
    zip(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖ",
        [
            "a",
            "be",
            "se",
            "de",
            "e",
            "eff",
            "ge",
            "hå",
            "i",
            "ji",
            "kå",
            "ell",
            "em",
            "en",
            "o",
            "pe",
            "kju",
            "err",
            "ess",
            "te",
            "u",
            "ve",
            "dubbel-ve",
            "eks",
            "y",
            "säta",
            "å",
            "ä",
            "ö",
        ],
    )
)

# A dot after one of these words is a full stop, not part of an abbreviation ("... t.ex. Jag ...").  The openers are
# capitalised: in raw text a sentence starts with a capital, while "t.ex. jag" is mid-sentence and keeps no dot.
_OPENERS = {
    "Jag",
    "Du",
    "Han",
    "Hon",
    "Vi",
    "Ni",
    "De",
    "Det",
    "Den",
    "Men",
    "Och",
    "Sedan",
    "Sen",
    "Då",
    "Nu",
    "Här",
    "Där",
    "Så",
    "Också",
    "Även",
    "Därför",
    "Eftersom",
    "Fast",
    "Eller",
    "När",
    "Om",
    "Att",
    "Idag",
    "Igår",
    "Imorgon",
}
_STOP = re.compile(rf"\s*$|\s+(?:{'|'.join(sorted(_OPENERS))})\b")

# Abbreviations, with their final dot.  Ordering by length keeps dotted forms ahead of their prefixes.
_ABBREVIATIONS = {
    "bl.a.": "bland annat",
    "bl. a.": "bland annat",
    "d.v.s.": "det vill säga",
    "dvs.": "det vill säga",
    "t.ex.": "till exempel",
    "t. ex.": "till exempel",
    "t.o.m.": "till och med",
    "t. o. m.": "till och med",
    "fr.o.m.": "från och med",
    "fr. o. m.": "från och med",
    "m.m.": "med mera",
    "m. m.": "med mera",
    "m.fl.": "med flera",
    "o.s.v.": "och så vidare",
    "osv.": "och så vidare",
    "etc.": "et cetera",
    "p.g.a.": "på grund av",
    "p. g. a.": "på grund av",
    "i.o.m.": "i och med",
    "s.k.": "så kallad",
    "s.k": "så kallad",
    "f.d.": "före detta",
    "f.d": "före detta",
    "ca.": "cirka",
    "kl.": "klockan",
    "nr.": "nummer",
    "n:o": "nummer",
    "sid.": "sida",
    "kap.": "kapitel",
    "avd.": "avdelning",
    "tel.": "telefon",
    "mob.": "mobil",
    "jfr.": "jämför",
    "resp.": "respektive",
    "inkl.": "inklusive",
    "exkl.": "exklusive",
    "max.": "maximalt",
    "ev.": "eventuellt",
    "ung.": "ungefär",
    "t.v.": "till vänster",
    "t.h.": "till höger",
}
_ABBREVIATION = re.compile(
    rf"(?<![a-zåäö])(?:{'|'.join(map(re.escape, sorted(_ABBREVIATIONS, key=len, reverse=True)))})", re.I
)
_TITLES = {
    "dr.": "doktor",
    "prof.": "professor",
    "docent.": "docent",
    "hr.": "herr",
    "frk.": "fröken",
    "s:t": "sankt",
    "st.": "sankt",
}
_TITLE = re.compile(
    rf"(?<![a-zåäö])(?i:{'|'.join(map(re.escape, sorted(_TITLES, key=len, reverse=True)))})(?=\s+[A-ZÅÄÖ])"
)
_MONTH_ABBREVIATION = re.compile(rf"(?<!\w)({'|'.join(sorted(_MONTH_ABBREVIATIONS, key=len, reverse=True))})\.", re.I)
_WEEKDAY_ABBREVIATION = re.compile(rf"(?<!\w)({'|'.join(_WEEKDAYS)})\.", re.I)
_NUMBER_WORD = re.compile(r"(?<!\w)(?:nr|n:o|no)\.?\s*(?=\d)", re.I)
# "5 st." is "fem stycken", not the title "sankt" (which only precedes a name); run before the title pass.
_STYCK = re.compile(r"(?<=\d)\s*st\.(?=\s|$)")

# "N." is not a Swedish ordinal; 1:a/1:e is.  Roman numerals after a name are ordinals (Carl XVI Gustaf).
_ORDINAL = re.compile(r"(?<![\w.,])(\d{1,6}):([ae])(?!\w)", re.I)
_ROMAN = re.compile(r"(?<![\w.])(?=[IVXL])((?:XL|L?X{0,3})(?:IX|IV|V?I{0,3}))(?![\w])")

_ACRONYM = re.compile(r"(?<!\w)[A-ZÅÄÖ][A-ZÅÄÖ0-9]{1,}(?!\w)")
_ACRONYM_WORDS = {"NATO", "IKEA", "UNESCO", "AIDS", "SAS", "SCA", "SEB", "ICA", "INTERPOL", "EUROPA"}
_SPECIAL_ACRONYMS = {
    "AB": "aktiebolag",
    "HB": "handelsbolag",
    "VD": "ve de",
    "EU": "e u",
    "USA": "u ess a",
    "TV": "te ve",
    "CD": "se de",
    "DVD": "de ve de",
    "SVT": "ess ve te",
    "SJ": "ess ji",
    "SR": "ess err",
    "FN": "eff en",
    "EKG": "e kå ge",
    "DNA": "de en a",
    "RNA": "err en a",
    "HIV": "hå i ve",
    "PDF": "pe de eff",
    "USB": "u ess be",
    "GPS": "ge pe ess",
    "SMS": "ess em ess",
    "MMS": "em em ess",
    "IT": "i te",
    "PC": "pe se",
    "AIK": "a i kå",
    "IFK": "i eff kå",
    "KTH": "kå te hå",
    "SCB": "ess se be",
    "MFF": "em eff eff",
    "ABB": "a be be",
    "OK": "okej",
}

# Currency: (one, singular, plural).  Swedish says "en krona", not "ett krona".
_CURRENCY = {
    "kr": ("en", "krona", "kronor"),
    "sek": ("en", "krona", "kronor"),
    ":-": ("en", "krona", "kronor"),
    "$": ("en", "dollar", "dollar"),
    "usd": ("en", "dollar", "dollar"),
    "€": ("en", "euro", "euro"),
    "eur": ("en", "euro", "euro"),
    "£": ("ett", "pund", "pund"),
    "gbp": ("ett", "pund", "pund"),
    "nok": ("en", "norsk krona", "norska kronor"),
    "dkk": ("en", "dansk krona", "danska kronor"),
}
_NUMBER_SRC = r"\d+(?:(?:[ \u00a0]\d{3})|(?:[.,]\d+))*"
_CURRENCY_AFTER = re.compile(
    rf"(?<![\w.,])({_NUMBER_SRC})\s*([$€£]|kr|sek|usd|eur|gbp|nok|dkk|:-)(?![a-zåäö])", re.I
)
_CURRENCY_BEFORE = re.compile(rf"([$€£])\s*({_NUMBER_SRC})")
_CURRENCY_WORD_BEFORE = re.compile(rf"(?<!\w)(kr|sek|usd|eur|gbp|nok|dkk)\s*({_NUMBER_SRC})", re.I)

# Units: (one, singular, plural).  Longest first in the pattern.
_UNITS = {
    "km/h": ("en", "kilometer i timmen", "kilometer i timmen"),
    "km/t": ("en", "kilometer i timmen", "kilometer i timmen"),
    "m/s": ("en", "meter per sekund", "meter per sekund"),
    "km²": ("en", "kvadratkilometer", "kvadratkilometer"),
    "km2": ("en", "kvadratkilometer", "kvadratkilometer"),
    "m²": ("en", "kvadratmeter", "kvadratmeter"),
    "m2": ("en", "kvadratmeter", "kvadratmeter"),
    "m³": ("en", "kubikmeter", "kubikmeter"),
    "m3": ("en", "kubikmeter", "kubikmeter"),
    "km": ("en", "kilometer", "kilometer"),
    "cm": ("en", "centimeter", "centimeter"),
    "mm": ("en", "millimeter", "millimeter"),
    "m": ("en", "meter", "meter"),
    "kg": ("ett", "kilogram", "kilogram"),
    "hg": ("ett", "hektogram", "hektogram"),
    "mg": ("ett", "milligram", "milligram"),
    "g": ("ett", "gram", "gram"),
    "dl": ("en", "deciliter", "deciliter"),
    "cl": ("en", "centiliter", "centiliter"),
    "ml": ("en", "milliliter", "milliliter"),
    "l": ("en", "liter", "liter"),
    "min": ("en", "minut", "minuter"),
    "sek": ("en", "sekund", "sekunder"),
    "ms": ("en", "millisekund", "millisekunder"),
    "s": ("en", "sekund", "sekunder"),
    "h": ("en", "timme", "timmar"),
    "°c": ("en", "grad celsius", "grader celsius"),
    "°f": ("en", "grad fahrenheit", "grader fahrenheit"),
    "°": ("en", "grad", "grader"),
    "gb": ("ett", "gigabyte", "gigabyte"),
    "mb": ("ett", "megabyte", "megabyte"),
    "kb": ("ett", "kilobyte", "kilobyte"),
    "tb": ("ett", "terabyte", "terabyte"),
    "st": ("ett", "styck", "stycken"),
    "mil": ("en", "mil", "mil"),
    "ton": ("ett", "ton", "ton"),
}
_UNIT = re.compile(
    rf"(?<![\w.,])({_NUMBER_SRC})\s*"
    rf"({'|'.join(map(re.escape, sorted(_UNITS, key=len, reverse=True)))})(?![a-zåäö0-9])",
    re.I,
)
_SPEED = re.compile(rf"(?<![\w.,])({_NUMBER_SRC})\s*km\s*/\s*(?:sa|s|h|t)(?![a-zåäö])", re.I)
_PERCENT = re.compile(rf"(?<![\w.,])({_NUMBER_SRC})\s*%")
_TIME = re.compile(r"(?<![\d:,.])(\d{1,2}):(\d{2})(?![\d:])")
_RANGE = re.compile(
    rf"(?<![\w.,])({_NUMBER_SRC})\s*[-–]\s*({_NUMBER_SRC})(?:\s*"
    rf"({'|'.join(map(re.escape, sorted(_UNITS, key=len, reverse=True)))}))?(?![a-zåäö0-9])",
    re.I,
)
_NEGATIVE = re.compile(r"(?<![\w.,])-(?=\d)")
_ISO_DATE = re.compile(r"(?<![\w/.-])(\d{4})-(\d{2})-(\d{2})(?![\w-])")
_ISO_SLASH_DATE = re.compile(r"(?<![\w/.-])(\d{4})/(\d{1,2})/(\d{1,2})(?![\w/])")
_SLASH_DATE = re.compile(r"(?<![\w/.-])(\d{1,2})[./](\d{1,2})[./](\d{4}|\d{2})(?![\w/])")
_DATE_WITH_MONTH = re.compile(rf"(?<!\w)(\d{{1,2}})\s+({'|'.join(_MONTHS)})(?:\s+(\d{{4}}))?(?!\w)", re.I)
_FRACTION = re.compile(r"(?<![\w/.,-])(?:(\d{1,3})\s+)?(\d{1,2})/(\d{1,3})(?![\w/])")
_DENOMINATORS = {
    2: ("halv", "halvor"),
    3: ("tredjedel", "tredjedelar"),
    4: ("fjärdedel", "fjärdedelar"),
    5: ("femtedel", "femtedelar"),
    6: ("sjättedel", "sjättedelar"),
    7: ("sjundedel", "sjundedelar"),
    8: ("åttondel", "åttondelar"),
    9: ("niondel", "niondelar"),
    10: ("tiondel", "tiondelar"),
    12: ("tolftedel", "tolftedelar"),
    16: ("sextondel", "sextondelar"),
    100: ("hundradel", "hundradelar"),
    1000: ("tusendel", "tusendelar"),
}

# Phone numbers: a leading 0 or +46/0046, 7-12 digits in groups.  Read digit by digit, group by group.
_PHONE = re.compile(r"(?<![\w.,+/-])(?:(?:\+46|0046)|0)[ -]?(?:\(?\d{1,4}\)?[ -]?){2,5}\d{2,4}(?![\d])")
_TELEPHONE = re.compile(r"(?<!\w)(?:tel[.:]?|telefon)\s*(?=[+(\d])", re.I)

# E-mail and web addresses are spelled out ("punkt", "snabel-a", "snedstreck").
_EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_DOMAIN_ENDINGS = "se|com|net|org|nu|io|eu|info|app|dev|edu|gov"
_URL = re.compile(
    r"(?<![\w@./-])(?:https?://[A-Za-z0-9.-]+|www\.[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
    rf"|[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:{_DOMAIN_ENDINGS})(?![A-Za-z0-9-]))(?:/[^\s]*)?"
)
_ADDRESS_CHARS = {
    ".": " punkt ",
    "/": " snedstreck ",
    "-": " streck ",
    "_": " understreck ",
    "@": " snabel-a ",
    ":": " ",
    "%": " procent ",
    "+": " plus ",
    "#": " nummer ",
}
_TRAILING = ".,!?;:)]}'\""
_CHAR_MAP = {
    "’": "'",
    "‘": "'",
    "ʼ": "'",
    "`": "'",
    "´": "'",
    "“": '"',
    "”": '"',
    "«": '"',
    "»": '"',
    "…": ".",
    "–": " ",
    "—": " ",
    "−": "-",
    ";": ",",
    ":": ",",
    "·": " ",
    "&": " och ",
    "+": " plus ",
    "=": " lika med ",
    "@": " snabel-a ",
    "#": " nummer ",
    "§": " paragraf ",
    "×": " gånger ",
    "÷": " delat med ",
    "±": " plus minus ",
    "≈": " ungefär ",
    "→": " till ",
    "≥": " minst ",
    "≤": " högst ",
    "½": " en halv ",
    "¼": " en fjärdedel ",
    "¾": " tre fjärdedelar ",
    "%": " procent ",
    "é": "e",
    "è": "e",
    "à": "a",
    "ä": "ä",
    "ö": "ö",
    "å": "å",
    "ü": "u",
    "ñ": "n",
    "ß": "ss",
    "ô": "o",
    "á": "a",
    "í": "i",
    "ó": "o",
    "ú": "u",
    "ç": "c",
}


def _below_thousand(n: int) -> str:
    """0-999 as one Swedish compound: 21 -> "tjugoett", 250 -> "tvåhundrafemtio"."""
    out = ""
    if n >= 100:
        hundreds, n = divmod(n, 100)
        out += ("" if hundreds == 1 else _ONES[hundreds]) + "hundra"
    if n >= 20:
        tens, n = divmod(n, 10)
        out += _TENS[tens] + (_ONES[n] if n else "")
    elif n >= 10:
        out += _TEENS[n]
    elif n:
        out += _ONES[n]
    return out


def number_to_words(n: int) -> str:
    """Cardinal number in Swedish: 1250 -> "ettusen tvåhundrafemtio", -3 -> "minus tre"."""
    if n < 0:
        return "minus " + number_to_words(-n)
    if n == 0:
        return "noll"
    if n >= 10**15:
        return " ".join(_ONES[int(d)] for d in str(n))
    parts: list[str] = []
    for value, singular, plural in _SCALES:
        if n >= value:
            q, n = divmod(n, value)
            parts.append(f"en {singular}" if q == 1 else f"{_below_thousand(q)} {plural}")
    if n >= 1000:
        q, n = divmod(n, 1000)
        parts.append("ettusen" if q == 1 else f"{_below_thousand(q)}tusen")
    if n:
        parts.append(_below_thousand(n))
    return " ".join(parts)


def year_to_words(n: int) -> str:
    """A four-digit year as Swedes read it: 1999 -> "nittonhundranittionio", 2024 -> "tvåtusentjugofyra"."""
    if 1100 <= n <= 1999:
        hi, lo = divmod(n, 100)
        return number_to_words(hi) + "hundra" + ("" if lo == 0 else _below_thousand(lo))
    if 2000 <= n <= 2099:
        return "tvåtusen" + ("" if n == 2000 else _below_thousand(n - 2000))
    return number_to_words(n)


def ordinal_to_words(n: int) -> str:
    """Swedish ordinal: 1 -> "första", 21 -> "tjugoförsta", 100 -> "hundrade"."""
    if n in _ORDINALS:
        return _ORDINALS[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ordinal_to_words(ones)
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return ("" if hundreds == 1 else _ONES[hundreds]) + "hundra" + ("de" if rest == 0 else ordinal_to_words(rest))
    if n < 10**6:
        q, rest = divmod(n, 1000)
        prefix = "ettusen" if q == 1 else f"{_below_thousand(q)}tusen"
        return prefix + ("de" if rest == 0 else ordinal_to_words(rest))
    for value, singular, plural in _SCALES:
        if n >= value:
            q, rest = divmod(n, value)
            prefix = f"en {singular}" if q == 1 else f"{_below_thousand(q)} {plural}"
            return prefix + ("te" if rest == 0 else " " + ordinal_to_words(rest))
    return number_to_words(n)


def _strip_thousands(s: str) -> str:
    s = s.replace("\u00a0", " ")
    if re.fullmatch(r"\d{1,3}(?:[ .]\d{3})+", s):
        return s.replace(" ", "").replace(".", "")
    return s


def _number_words(raw: str, year: bool = True) -> str:
    """Verbalise a numeric token, including decimal commas and leading zeros."""
    raw = raw.strip().replace("\u00a0", " ")
    if raw.startswith("-"):
        return "minus " + _number_words(raw[1:], year)
    if raw.count(",") > 1:  # English-style thousands ("1,250,000"): not a decimal comma
        raw = raw.replace(",", "")
    # Decimal comma: 3,14 -> "tre komma ett fyra" (digits are safer than inventing a denominator).
    if "," in raw:
        whole, frac = raw.split(",", 1)
        whole = _strip_thousands(whole)
        whole_words = number_to_words(int(whole)) if whole else "noll"
        return f"{whole_words} komma " + " ".join(_ONES[int(d)] for d in frac)
    compact = _strip_thousands(raw)
    if re.fullmatch(r"\d+\.\d+", compact):  # an English-style decimal
        whole, frac = compact.split(".", 1)
        return f"{number_to_words(int(whole))} komma " + " ".join(_ONES[int(d)] for d in frac)
    if len(compact) == 4 and year and 1100 <= int(compact) <= 2099:
        return year_to_words(int(compact))
    if len(compact) > 1 and compact[0] == "0":
        return " ".join(_ONES[int(d)] for d in compact)
    try:
        return number_to_words(int(compact))
    except ValueError:  # malformed grouping ("1.2.3"): safest is digit by digit
        return " ".join(_ONES[int(d)] for d in compact if d.isdigit())


def _value(raw: str) -> float | None:
    s = _strip_thousands(raw.replace(",", "."))
    try:
        return float(s)
    except ValueError:
        return None


def _one_or_many(raw: str, one: str, singular: str, plural: str, year: bool = True) -> str:
    return f"{one} {singular}" if _value(raw) == 1 else f"{_number_words(raw, year)} {plural}"


def _verbalise_time(m: re.Match) -> str:
    hour, minute = int(m.group(1)), int(m.group(2))
    if hour > 24 or minute > 59:
        return f" {number_to_words(hour)} {minute:02d} "
    words = number_to_words(hour)
    if minute:
        words += " noll " + _ONES[minute] if minute < 10 else " " + number_to_words(minute)
    return f" {words} "


def _verbalise_phone(m: re.Match) -> str:
    digits = "".join(c for c in m.group() if c.isdigit())
    if not 7 <= len(digits) <= 13:
        return m.group()
    prefix = ""
    if digits.startswith("0046"):
        prefix, digits = "plus fyra sex ", digits[4:]
    elif digits.startswith("46") and m.group().startswith("+"):
        prefix, digits = "plus fyra sex ", digits[2:]
    groups = re.findall(r"\d+", digits)
    spoken = [prefix + " ".join(_ONES[int(d)] for d in groups[0])]
    spoken += [" ".join(_ONES[int(d)] for d in group) for group in groups[1:]]
    return f" {', '.join(spoken)} "


def _verbalise_date(day: int, month: int, year: str | None) -> str | None:
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    words = f"{ordinal_to_words(day)} {_MONTHS[month - 1]}"
    if year:
        words += " " + (year_to_words(int(year)) if len(year) == 4 else number_to_words(int(year)))
    return f" {words} "


def _slash_date(m: re.Match) -> str:
    a, b, year = int(m.group(1)), int(m.group(2)), m.group(3)
    # Swedish dates are day-first: 3/4/2021 is 3 April (swap only when the first number cannot be a month).
    return _verbalise_date(a, b, year) or _verbalise_date(b, a, year) or m.group()


def _iso_date(m: re.Match) -> str:
    return _verbalise_date(int(m.group(3)), int(m.group(2)), m.group(1)) or m.group()


def _iso_slash_date(m: re.Match) -> str:
    return _verbalise_date(int(m.group(3)), int(m.group(2)), m.group(1)) or m.group()


def _date_with_month(m: re.Match) -> str:
    month = _MONTHS.index(m.group(2).lower()) + 1
    return _verbalise_date(int(m.group(1)), month, m.group(3)) or m.group()


def _fraction(m: re.Match) -> str:
    whole, numerator, denominator = m.group(1), int(m.group(2)), int(m.group(3))
    if denominator not in _DENOMINATORS or not 0 < numerator <= denominator:
        return m.group()
    one, many = _DENOMINATORS[denominator]
    part = one if numerator == 1 else many
    head = "" if whole is None else f"{'en' if int(whole) == 1 else number_to_words(int(whole))} och "
    return f" {head}{'en' if numerator == 1 else number_to_words(numerator)} {part} "


def _range(m: re.Match) -> str:
    first, second, unit = m.group(1), m.group(2), m.group(3)
    words = f"{_number_words(first)} till {_number_words(second)}"
    if unit:
        one, singular, plural = _UNITS[unit.lower()]
        words += " " + (singular if _value(second) == 1 else plural)
    return f" {words} "


def _currency(m: re.Match, before: bool = False) -> str:
    if before:
        symbol, amount = m.group(1).lower(), m.group(2)
    else:
        amount, symbol = m.group(1), m.group(2).lower()
    one, singular, plural = _CURRENCY[symbol]
    return f" {_one_or_many(amount, one, singular, plural, year=False)} "


def _unit(m: re.Match) -> str:
    amount, unit = m.group(1), m.group(2).lower()
    one, singular, plural = _UNITS[unit]
    return f" {_one_or_many(amount, one, singular, plural, year=False)} "


def _percent(m: re.Match) -> str:
    amount = m.group(1)
    return f" {'en procent' if _value(amount) == 1 else f'{_number_words(amount, year=False)} procent'} "


def _abbreviation(m: re.Match) -> str:
    expansion = _ABBREVIATIONS[m.group(0).lower()]
    return expansion + ("." if _STOP.match(m.string, m.end()) else "")


def _title(m: re.Match) -> str:
    return _TITLES[m.group(0).lower()] + " "


def _month_abbreviation(m: re.Match) -> str:
    return _MONTH_ABBREVIATIONS[m.group(1).lower()] + " "


def _weekday_abbreviation(m: re.Match) -> str:
    return _WEEKDAYS[m.group(1).lower()] + " "


def _roman_to_int(s: str) -> int:
    values = [{"I": 1, "V": 5, "X": 10, "L": 50}[c] for c in s]
    return sum(-v if v < nxt else v for v, nxt in zip(values, [*values[1:], 0]))


def _roman(m: re.Match) -> str:
    numeral = m.group(1)
    if len(numeral) < 2:  # a lone I/V/X is not a regnal number
        return m.group()
    return f" {ordinal_to_words(_roman_to_int(numeral))} "


def _spell(letters: str) -> str:
    return " ".join(_LETTER_NAMES[c] for c in letters if c in _LETTER_NAMES)


def _acronym(m: re.Match) -> str:
    word = m.group()
    if word in _SPECIAL_ACRONYMS:
        return _SPECIAL_ACRONYMS[word]
    if word in _ACRONYM_WORDS:
        return word.lower()
    if not re.search(r"[AEIOUYÅÄÖ]", word) or len(word) <= 2:
        return _spell(word)
    return word.lower()


def _speak_address(address: str) -> str:
    """An e-mail or web address without its scheme, query and fragment: dots, slashes and dashes by name."""
    address = re.split("[?#]", re.sub("^https?://", "", address), maxsplit=1)[0].rstrip("/")
    www, address = ("ve ve ve punkt ", address[4:]) if address.lower().startswith("www.") else ("", address)
    return www + "".join(_ADDRESS_CHARS.get(c, c) for c in address)


def _address(m: re.Match) -> str:
    address = m.group().rstrip(_TRAILING)  # punctuation after the address belongs to the sentence
    tail = m.group()[len(address) :]
    return f" {_speak_address(address)}{tail} "


def _verbalise_number(m: re.Match) -> str:
    return f" {_number_words(m.group(), year=True)} "


def normalize(text: str) -> str:
    """Normalise raw Swedish text to lower-case orthographic Swedish (the input of :func:`phonemize`).

    Handles numbers, clock times, money, percentages, units, dates, fractions, phone numbers, common abbreviations,
    titles, acronyms, e-mail and web addresses.  Only a sentence end keeps its dot; :func:`split_sentences` splits
    there.
    """
    text = unicodedata.normalize("NFC", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Addresses and phone numbers before punctuation is rewritten.
    text = _EMAIL.sub(_address, text)
    text = _URL.sub(_address, text)
    text = _TELEPHONE.sub("telefon ", text)
    text = _PHONE.sub(_verbalise_phone, text)
    # Month and weekday abbreviations first, so "5 jan. 2020" becomes a full date and is read as one.
    text = _STYCK.sub(" stycken ", text)
    text = _MONTH_ABBREVIATION.sub(_month_abbreviation, text)
    text = _WEEKDAY_ABBREVIATION.sub(_weekday_abbreviation, text)
    # Structured dates and times before their separators are mapped to spaces or commas.
    text = _ISO_DATE.sub(_iso_date, text)
    text = _ISO_SLASH_DATE.sub(_iso_slash_date, text)
    text = _DATE_WITH_MONTH.sub(_date_with_month, text)
    text = _SLASH_DATE.sub(_slash_date, text)
    text = _TIME.sub(_verbalise_time, text)
    # Abbreviations and titles (their dots must be resolved before the generic punctuation pass).
    text = _TITLE.sub(_title, text)
    text = _ABBREVIATION.sub(_abbreviation, text)
    text = _NUMBER_WORD.sub("nummer ", text)
    text = _ROMAN.sub(_roman, text)
    # Money, units, percentages and ranges before bare numbers.
    text = _CURRENCY_BEFORE.sub(lambda m: _currency(m, before=True), text)
    text = _CURRENCY_WORD_BEFORE.sub(lambda m: _currency(m, before=True), text)
    text = _CURRENCY_AFTER.sub(_currency, text)
    text = _PERCENT.sub(_percent, text)
    text = _SPEED.sub(
        lambda m: f" {_one_or_many(m.group(1), 'en', 'kilometer i timmen', 'kilometer i timmen', year=False)} ", text
    )
    text = _RANGE.sub(_range, text)
    text = _UNIT.sub(_unit, text)
    text = _ORDINAL.sub(lambda m: f" {ordinal_to_words(int(m.group(1)))} ", text)
    text = _FRACTION.sub(_fraction, text)
    text = "".join(_CHAR_MAP.get(c, c) for c in text)
    text = _NEGATIVE.sub(" minus ", text)
    text = _ACRONYM.sub(_acronym, text)
    text = re.sub(r"\d+(?:(?:[ \u00a0]\d{3})|(?:[.,]\d+))*", _verbalise_number, text)
    text = text.lower()
    # Keep Swedish letters, sentence punctuation and hyphens (compounds such as "e-post").
    text = re.sub(r"[^a-zåäöé0-9\s.,!?\-]", " ", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+([.,!?])", r"\1", text)  # no space before punctuation
    text = re.sub(r"([.,!?])(?=[^\s.,!?])", r"\1 ", text)
    text = re.sub(r",+", ",", text)
    text = re.sub(r",(?=[.!?])", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[.,!? ]+", "", text)
    return text


def _phonemize(text: str) -> str:
    """IPA of already normalised Swedish text, through phonemizer and espeak-ng."""
    try:
        from phonemizer import phonemize as _p
        from phonemizer.separator import Separator
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Swedish TTS needs the optional phonemizer package and the espeak-ng system library: "
            "`pip install phonemizer` and `brew install espeak-ng` (macOS) or `apt install espeak-ng` (Debian/Ubuntu)."
        ) from e
    try:
        out = _p(
            [text],
            language=ESPEAK_VOICE,
            backend="espeak",
            with_stress=True,
            preserve_punctuation=True,
            strip=True,
            separator=Separator(phone="", word=" ", syllable=""),
        )[0]
    except Exception as e:  # pragma: no cover - missing shared library / voice
        raise RuntimeError(
            "espeak-ng could not phonemise Swedish; check that the `espeak-ng` binary and its data are installed "
            "and that the `sv` voice is available (`espeak-ng --voices=sv`)."
        ) from e
    # espeak-ng 1.52 writes the Swedish sj phoneme as the two-character name "sx" (sometimes with a zero-width
    # joiner); it is one sound in Swedish, so it gets one symbol.
    out = out.replace("\u200d", "").replace("sx", "ɧ")
    out = "".join(c if c in SYMBOL_TO_ID else " " for c in out)
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"\s+([.,!?])", r"\1", out)
    out = re.sub(r"([.,!?])(?=[^\s.,!?])", r"\1 ", out)
    return out


def phonemize(text: str, normalized: bool = False) -> str:
    """Raw or normalised Swedish -> IPA.  ``normalized=True`` skips :func:`normalize`."""
    return _phonemize(text if normalized else normalize(text))


def text_to_ids(
    text: str, intersperse_blank: bool = True, normalized: bool = False, phonemized: bool = False
) -> list[int]:
    """Map text to phoneme ids; optionally intersperse a blank token (Grad-TTS) for alignment.

    ``normalized=True`` means the text has already passed through :func:`normalize`; it is still phonemised.
    ``phonemized=True`` maps an IPA string directly (e.g. from :func:`phonemize`)."""
    if not phonemized:
        text = phonemize(text, normalized=normalized)
    ids = [SYMBOL_TO_ID[c] for c in text]
    if intersperse_blank:
        out = [BLANK_ID] * (2 * len(ids) + 1)
        out[1::2] = ids
        ids = out
    return ids


def ids_to_text(ids: list[int]) -> str:
    return "".join(SYMBOLS[i] for i in ids if i not in (PAD_ID, BLANK_ID))
