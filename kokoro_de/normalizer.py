"""
Lightweight German text normalization used before routing into G2P.

This is intentionally conservative. It handles a small set of high-value cases
locally so the repo does not depend on an external German-normalization fork.
"""

from __future__ import annotations

import re

_MONTHS = {
    1: "januar",
    2: "februar",
    3: "märz",
    4: "april",
    5: "mai",
    6: "juni",
    7: "juli",
    8: "august",
    9: "september",
    10: "oktober",
    11: "november",
    12: "dezember",
}

_ABBREVIATIONS = {
    r"\bz\.\s?B\.": "zum beispiel",
    r"\bu\.\s?a\.": "unter anderem",
    r"\busw\.": "und so weiter",
    r"\bca\.": "circa",
    r"\bDr\.": "Doktor",
    r"\bProf\.": "Professor",
    r"\bGmbH\b": "Gesellschaft mit beschränkter Haftung",
}
_LETTER_ACRONYMS = {
    "AI": "a i",
    "API": "a p i",
    "CPU": "c p u",
    "GPU": "g p u",
    "KI": "k i",
    "LLM": "l l m",
    "NLP": "n l p",
    "TTS": "t t s",
    "UI": "u i",
    "UX": "u x",
}

_DATE_WITH_AM_RE = re.compile(r"\bam\s+(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b", flags=re.IGNORECASE)
_DATE_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b")
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_EURO_RE = re.compile(r"(?<!\w)(\d+(?:[.,]\d{1,2})?)\s*€|€\s*(\d+(?:[.,]\d{1,2})?)")
_DECIMAL_RE = re.compile(r"\b(\d+),(\d+)\b")
_PERCENT_RE = re.compile(r"\b(\d+(?:,\d+)?)\s*%")
_UNIT_RE = re.compile(
    r"\b(\d+(?:,\d+)?)\s*(km|kg|cm|mm|khz|mhz|ghz|hz|gb|tb|mb|ms|s|m|°c)\b",
    flags=re.IGNORECASE,
)
_ORDINAL_RE = re.compile(r"\b(\d+)\.")
_ACRONYM_RE = re.compile(r"\b(?:AI|API|CPU|GPU|KI|LLM|NLP|TTS|UI|UX)\b")
_VERSION_RE = re.compile(r"\bv(\d+)(?:\.(\d+))?\b", flags=re.IGNORECASE)
_INT_RE = re.compile(r"\b\d+\b")

_UNITS = {
    0: "null",
    1: "eins",
    2: "zwei",
    3: "drei",
    4: "vier",
    5: "fünf",
    6: "sechs",
    7: "sieben",
    8: "acht",
    9: "neun",
    10: "zehn",
    11: "elf",
    12: "zwölf",
    13: "dreizehn",
    14: "vierzehn",
    15: "fünfzehn",
    16: "sechzehn",
    17: "siebzehn",
    18: "achtzehn",
    19: "neunzehn",
}

_TENS = {
    20: "zwanzig",
    30: "dreißig",
    40: "vierzig",
    50: "fünfzig",
    60: "sechzig",
    70: "siebzig",
    80: "achtzig",
    90: "neunzig",
}

_ORDINAL_BASE = {
    1: "erste",
    2: "zweite",
    3: "dritte",
    4: "vierte",
    5: "fünfte",
    6: "sechste",
    7: "siebte",
    8: "achte",
    9: "neunte",
    10: "zehnte",
    11: "elfte",
    12: "zwölfte",
    13: "dreizehnte",
    14: "vierzehnte",
    15: "fünfzehnte",
    16: "sechzehnte",
    17: "siebzehnte",
    18: "achtzehnte",
    19: "neunzehnte",
}


def _unit_word(value: int, compound: bool = False) -> str:
    if value == 1:
        return "ein" if compound else "eins"
    return _UNITS[value]


def number_to_de(value: int) -> str:
    if value < 0:
        return "minus " + number_to_de(-value)
    if value < 20:
        return _UNITS[value]
    if value < 100:
        ones = value % 10
        tens = value - ones
        if ones == 0:
            return _TENS[tens]
        return f"{_unit_word(ones, compound=True)}und{_TENS[tens]}"
    if value < 1000:
        hundreds = value // 100
        rest = value % 100
        head = "einhundert" if hundreds == 1 else f"{_unit_word(hundreds, compound=True)}hundert"
        return head if rest == 0 else head + number_to_de(rest)
    if value < 1_000_000:
        thousands = value // 1000
        rest = value % 1000
        head = "eintausend" if thousands == 1 else f"{number_to_de(thousands)}tausend"
        return head if rest == 0 else head + number_to_de(rest)
    return str(value)


def ordinal_to_de(day: int) -> str:
    if day in _ORDINAL_BASE:
        return _ORDINAL_BASE[day]
    if day < 20:
        return number_to_de(day) + "te"
    return number_to_de(day) + "ste"


def ordinal_date_dative(day: int) -> str:
    base = ordinal_to_de(day)
    if base.endswith("e"):
        return base[:-1] + "en"
    return base + "n"


def _replace_date_parts(day: int, month: int, year: int, *, dative: bool) -> str:
    month_name = _MONTHS.get(month)
    if month_name is None:
        return f"{day}.{month}.{year}"
    ordinal = ordinal_date_dative(day) if dative else ordinal_to_de(day)
    return f"{ordinal} {month_name} {number_to_de(year)}"


def _replace_date_with_am(match: re.Match[str]) -> str:
    day = int(match.group(1))
    month = int(match.group(2))
    year = int(match.group(3))
    return "am " + _replace_date_parts(day, month, year, dative=True)


def _replace_date(match: re.Match[str]) -> str:
    day = int(match.group(1))
    month = int(match.group(2))
    year = int(match.group(3))
    return _replace_date_parts(day, month, year, dative=False)


def _replace_time(match: re.Match[str]) -> str:
    hours = int(match.group(1))
    minutes = int(match.group(2))
    if minutes == 0:
        return f"{number_to_de(hours)} uhr"
    return f"{number_to_de(hours)} uhr {number_to_de(minutes)}"


def _replace_euro(match: re.Match[str]) -> str:
    raw = match.group(1) or match.group(2) or ""
    normalized = raw.replace(",", ".")
    if "." in normalized:
        euros_raw, cents_raw = normalized.split(".", 1)
        euros = int(euros_raw)
        cents = int((cents_raw + "0")[:2])
    else:
        euros = int(normalized)
        cents = 0
    result = f"{number_to_de(euros)} euro"
    if cents:
        result += f" und {number_to_de(cents)} cent"
    return result


def _spell_digits(value: str) -> str:
    return " ".join(number_to_de(int(char)) for char in value if char.isdigit())


def _replace_decimal(match: re.Match[str]) -> str:
    whole = int(match.group(1))
    frac = match.group(2)
    return f"{number_to_de(whole)} komma {_spell_digits(frac)}"


def _replace_percent(match: re.Match[str]) -> str:
    number = _replace_decimal(re.match(r"(\d+),(\d+)", match.group(1))) if "," in match.group(1) else number_to_de(int(match.group(1)))
    return f"{number} prozent"


def _replace_unit(match: re.Match[str]) -> str:
    number_raw = match.group(1)
    unit_raw = match.group(2).lower()
    number = _replace_decimal(re.match(r"(\d+),(\d+)", number_raw)) if "," in number_raw else number_to_de(int(number_raw))
    units = {
        "km": "kilometer",
        "kg": "kilogramm",
        "cm": "zentimeter",
        "mm": "millimeter",
        "hz": "hertz",
        "khz": "kilohertz",
        "mhz": "megahertz",
        "ghz": "gigahertz",
        "gb": "gigabyte",
        "tb": "terabyte",
        "mb": "megabyte",
        "ms": "millisekunden",
        "s": "sekunden",
        "m": "meter",
        "°c": "grad celsius",
    }
    return f"{number} {units[unit_raw]}"


def _replace_ordinal(match: re.Match[str]) -> str:
    value = int(match.group(1))
    return ordinal_to_de(value)


def _replace_acronym(match: re.Match[str]) -> str:
    return _LETTER_ACRONYMS[match.group(0)]


def _replace_version(match: re.Match[str]) -> str:
    major = number_to_de(int(match.group(1)))
    minor = match.group(2)
    if minor is None:
        return f"version {major}"
    return f"version {major} punkt {_spell_digits(minor)}"


def _replace_int(match: re.Match[str]) -> str:
    raw = match.group(0)
    try:
        value = int(raw)
    except ValueError:
        return raw
    if value >= 1_000_000:
        return raw
    return number_to_de(value)


def normalize_text_de(text: str) -> str:
    normalized = text
    for pattern, replacement in _ABBREVIATIONS.items():
        normalized = re.sub(pattern, replacement, normalized)
    normalized = _DATE_WITH_AM_RE.sub(_replace_date_with_am, normalized)
    normalized = _DATE_RE.sub(_replace_date, normalized)
    normalized = _TIME_RE.sub(_replace_time, normalized)
    normalized = _EURO_RE.sub(_replace_euro, normalized)
    normalized = _PERCENT_RE.sub(_replace_percent, normalized)
    normalized = _UNIT_RE.sub(_replace_unit, normalized)
    normalized = _DECIMAL_RE.sub(_replace_decimal, normalized)
    normalized = _VERSION_RE.sub(_replace_version, normalized)
    normalized = _ORDINAL_RE.sub(_replace_ordinal, normalized)
    normalized = _ACRONYM_RE.sub(_replace_acronym, normalized)
    normalized = _INT_RE.sub(_replace_int, normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized
