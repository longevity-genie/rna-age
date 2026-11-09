"""Parse donor age from free-text sample annotations (GEO ``characteristics_ch1``, BioSample attributes, ...).

:func:`parse_age_field` takes one ``key: value`` pair and returns an :class:`Age` or None.
:func:`find_age` scans all fields of a sample and returns the best age found.

Rules:
- The key must name the donor's age (``age``, ``age (years)``, ``donor age``, ``age at diagnosis``,
  ``age at death``...). Keys about other clocks are rejected: gestational age, passage, time post infection,
  age at onset, differentiation day, developmental stage.
- Units come from the value (``54 years``, ``6 months``, ``3 wk``, ``10 days``) or else from the key
  (``age (months): 6``); default unit is years. Months, weeks and days are converted to years.
- ``50-59``, ``50 to 59``, ``30s`` → binned; ``>89``, ``90+``, ``≥90`` → censored with ``age_min``; ``<1`` → binned
  [0, 0].
- Non-ages → None: ``NA``, ``unknown``, ``adult``, ``young``, words in general, prenatal values (gestation weeks,
  ``E14.5``, ``pcw``), values outside 0–120 years.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DAYS_PER_YEAR = 365.25
WEEKS_PER_YEAR = DAYS_PER_YEAR / 7


@dataclass(frozen=True)
class Age:
    age_type: str  # exact | binned | censored
    age_years: float | None
    age_min: float | None
    age_max: float | None
    raw: str  # "key: value" as found
    key_kind: str = "age"  # age | diagnosis | death | sampling | enrollment


NA_VALUES = {
    "",
    "na",
    "n/a",
    "n.a.",
    "nan",
    "none",
    "null",
    "unknown",
    "unk",
    "not available",
    "not applicable",
    "not determined",
    "not provided",
    "not collected",
    "missing",
    "-",
    "--",
    "?",
    "nd",
    "not reported",
    "unavailable",
    "not recorded",
    "restricted access",
    "withheld",
}

# keys that mention "age" but are not the donor's chronological age at sampling
_REJECT_KEY = re.compile(
    r"gestation|gestational|embryo|fetal|foetal|\bpcw\b|post[- ]?concep|passage|onset|infect|\bdpi\b|\bhpi\b|"
    r"\bhpf\b|post[- ]?(?:natal|partum|op|transplant|injury|treatment|vaccin)|differentiat|culture|stage|"
    r"\bdays? in\b|^time|time ?point|time post|average|percentage|\bpage\b|dosage|cleavage|lineage|leakage|storage|"
    r"damage|drainage|package|coverage|mileage|voltage|image|language|usage|\bwage|\bsage\b|\bmenarche|menopause|"
    r"gender|tissue|cell|mother|maternal|father|paternal|parent|\bat (?:first|last) (?:birth|pregnan)|"
    r"smoking|pack|\bhrt\b|ages? of (?:the )?(?:mother|father)|sample age|\brin\b|storage|bone age|brain age|"
    r"biological age|epigenetic age|predicted|clock",
    re.I,
)
_AGE_KEY = re.compile(r"(?:^|[\s_(/.-])age(?:$|[\s_):/.-]|s\b|yrs?\b|years?\b|months?\b|_?at\b)|^age", re.I)

_KEY_KIND = (
    (re.compile(r"diagnos", re.I), "diagnosis"),
    (re.compile(r"death|deceased|autopsy", re.I), "death"),
    (re.compile(r"enrol|baseline|recruit|entry|consent", re.I), "enrollment"),
    (re.compile(r"sampl|collect|biops|surgery|draw|visit|operation|harvest|donat", re.I), "sampling"),
)

_UNIT = {
    "y": 1.0,
    "yr": 1.0,
    "yrs": 1.0,
    "year": 1.0,
    "years": 1.0,
    "yo": 1.0,
    "y.o.": 1.0,
    "y.o": 1.0,
    "a": 1.0,
    "years old": 1.0,
    "year old": 1.0,
    "-year-old": 1.0,
    "year-old": 1.0,
    "yr old": 1.0,
    "yrs old": 1.0,
    "m": 1 / 12,
    "mo": 1 / 12,
    "mos": 1 / 12,
    "month": 1 / 12,
    "months": 1 / 12,
    "mth": 1 / 12,
    "mths": 1 / 12,
    "months old": 1 / 12,
    "month old": 1 / 12,
    "w": 1 / WEEKS_PER_YEAR,
    "wk": 1 / WEEKS_PER_YEAR,
    "wks": 1 / WEEKS_PER_YEAR,
    "week": 1 / WEEKS_PER_YEAR,
    "weeks": 1 / WEEKS_PER_YEAR,
    "weeks old": 1 / WEEKS_PER_YEAR,
    "d": 1 / DAYS_PER_YEAR,
    "day": 1 / DAYS_PER_YEAR,
    "days": 1 / DAYS_PER_YEAR,
    "days old": 1 / DAYS_PER_YEAR,
}
_KEY_UNIT = (
    (re.compile(r"month|\bmo\b|\(m\)|_m$", re.I), 1 / 12),
    (re.compile(r"week|\bwk", re.I), 1 / WEEKS_PER_YEAR),
    (re.compile(r"\bdays?\b|\(d\)|_days?\b|in days", re.I), 1 / DAYS_PER_YEAR),
)
_NUM = r"(\d+(?:\.\d+)?)"
_UNITS_RE = "|".join(sorted((re.escape(u) for u in _UNIT), key=len, reverse=True))
_SINGLE = re.compile(rf"^(?:about |approx\.? |~|ca\.? )?{_NUM}\s*({_UNITS_RE})?\.?$", re.I)
_COMPOUND = re.compile(
    rf"^{_NUM}\s*(y|yr|yrs|years?)[,\s]*(?:and\s*)?{_NUM}\s*(m|mo|mos|months?)(?:[,\s]*{_NUM}\s*(d|days?))?$", re.I
)
_RANGE = re.compile(rf"^{_NUM}\s*(?:-|–|—|to|~)\s*{_NUM}\s*({_UNITS_RE})?$", re.I)
_DECADE = re.compile(r"^(\d)0'?s$", re.I)
_CENSOR_UP = re.compile(
    rf"^(?:>\s*=|≥|>|over|older than|above)\s*{_NUM}\s*({_UNITS_RE})?$|^{_NUM}\s*(?:\+|or older|and older|and above|or above|\s*plus)\s*({_UNITS_RE})?$",
    re.I,
)
_CENSOR_DOWN = re.compile(rf"^(?:<\s*=|≤|<|under|younger than|below)\s*{_NUM}\s*({_UNITS_RE})?$", re.I)
_PRENATAL = re.compile(
    r"gestation|\bgw\b|\bpcw\b|post[- ]?concep|\bfetal\b|\bfoetal\b|\bfetus\b|embryo|^e\d+(?:\.\d+)?$|"
    r"carnegie|trimester|prenatal|weeks? pc\b",
    re.I,
)
_TIME_UNITS = re.compile(r"\b(?:dpi|hpi|hpf|dpf|hours?|hrs?|h|min(?:utes)?|passages?|p\d+|dpc)\b", re.I)


def is_age_key(key: str) -> bool:
    k = key.strip()
    return bool(_AGE_KEY.search(k)) and not _REJECT_KEY.search(k)


def key_kind(key: str) -> str:
    for rx, kind in _KEY_KIND:
        if rx.search(key):
            return kind
    return "age"


def _unit(value_unit: str | None, key: str) -> float:
    if value_unit:
        return _UNIT[value_unit.lower().rstrip(".") if value_unit.lower().rstrip(".") in _UNIT else value_unit.lower()]
    for rx, factor in _KEY_UNIT:
        if rx.search(key):
            return factor
    return 1.0


def _plausible(years: float) -> bool:
    return 0 <= years <= 120


def looks_prenatal(text: str) -> bool:
    return bool(_PRENATAL.search(text))


def parse_age_value(value: str, key: str = "age") -> Age | None:
    """Parse the value of an age field (the key supplies the unit when the value has none)."""
    raw = f"{key}: {value}"
    v = value.strip().strip(";,").strip()
    v = re.sub(r"\s+", " ", v)
    low = v.lower()
    if low in NA_VALUES or looks_prenatal(low) or _TIME_UNITS.search(low):
        return None
    kind = key_kind(key)
    low = re.sub(r"^(?:age[d:]?\s*)", "", low)  # "aged 54", "age 54" inside the value
    low = low.replace(" years of age", " years").replace(" of age", "")

    if m := _SINGLE.match(low):
        years = float(m.group(1)) * _unit(m.group(2), key)
        return Age("exact", years, years, years, raw, kind) if _plausible(years) else None
    if m := _COMPOUND.match(low):
        years = float(m.group(1)) + float(m.group(3)) / 12 + (float(m.group(5)) / DAYS_PER_YEAR if m.group(5) else 0)
        return Age("exact", years, years, years, raw, kind) if _plausible(years) else None
    if m := _RANGE.match(low):
        f = _unit(m.group(3), key)
        lo, hi = float(m.group(1)) * f, float(m.group(2)) * f
        if lo > hi or not (_plausible(lo) and _plausible(hi)):
            return None
        if lo == hi:
            return Age("exact", lo, lo, lo, raw, kind)
        return Age("binned", None, lo, hi, raw, kind)
    if m := _DECADE.match(low):
        lo = float(m.group(1)) * 10
        return Age("binned", None, lo, lo + 9, raw, kind)
    if m := _CENSOR_UP.match(low):
        num, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        f = _unit(unit, key)
        strict = low.startswith((">", "over", "older", "above")) and not low.startswith((">=", "> ="))
        lo = float(num) * f + (1 if strict and f == 1.0 else 0)
        return Age("censored", None, lo, None, raw, kind) if _plausible(lo) else None
    if m := _CENSOR_DOWN.match(low):
        f = _unit(m.group(2), key)
        hi = float(m.group(1)) * f
        if not _plausible(hi):
            return None
        strict = low.startswith(("<", "under", "younger", "below")) and not low.startswith(("<=", "< ="))
        return Age("binned", None, 0.0, max(0.0, hi - 1) if strict and f == 1.0 else hi, raw, kind)
    return None


def parse_age_field(key: str, value: str) -> Age | None:
    """Parse one ``key: value`` annotation; None when the key is not a donor age or the value is not an age."""
    if not is_age_key(key):
        return None
    return parse_age_value(value, key)


_KIND_RANK = {"sampling": 0, "age": 1, "enrollment": 2, "diagnosis": 3, "death": 4}
_TYPE_RANK = {"exact": 0, "censored": 1, "binned": 2}


def find_age(fields: list[tuple[str, str]]) -> Age | None:
    """Best age among a sample's annotation fields: exact before censored before binned, then by key kind."""
    found = [a for k, v in fields if (a := parse_age_field(k, v)) is not None]
    if not found:
        return None
    return min(found, key=lambda a: (_TYPE_RANK[a.age_type], _KIND_RANK[a.key_kind]))


def split_characteristics(text: str, sep: str = "\t") -> list[tuple[str, str]]:
    """Split a characteristics string (``"key: value<sep>key: value"``) into pairs; items without a colon get key ""."""
    out = []
    for item in text.split(sep):
        item = item.strip()
        if not item:
            continue
        key, colon, value = item.partition(":")
        out.append((key.strip(), value.strip()) if colon else ("", item))
    return out
