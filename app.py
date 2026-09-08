"""
Shorthand Word Lookup Web App
=============================

Local Flask web app for studying English word frequency, IPA pronunciation,
grammatical classification, Major system values, and Gregg shorthand SVGs.

Supports single-word lookup, phrase lookup, segment-mode SVG filename
matching, and outline-only live suggestions while the user types.

Examples:
    jewel      -> media/gregg/jewel.svg
    this is    -> media/gregg/this_is.svg
    some of    -> media/gregg/some_of.svg
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import sqlite3
import unicodedata

import nltk
from nltk.corpus import wordnet as wn

from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from wordfreq import zipf_frequency
import eng_to_ipa as ipa

app = Flask(__name__)

SVG_DIR = (Path(__file__).parent / "media" / "gregg").resolve()


# ---------------------------------------------------------------------------
# Word and phrase normalization / validation
# ---------------------------------------------------------------------------


def is_acceptable_word(raw_word: str) -> bool:
    """
    Validate user input before lookup.

    Accepted:
    - Single words
    - Multi-word phrases
    - Letters a-z
    - Optional internal apostrophes
    - Optional internal hyphens
    - Single spaces between words
    """

    cleaned = raw_word.strip().lower()

    if not cleaned:
        return False

    word_pattern = r"[a-z]+(?:['-][a-z]+)*"
    phrase_pattern = rf"^{word_pattern}(?: {word_pattern})*$"

    return re.fullmatch(phrase_pattern, cleaned) is not None


def normalize_word(text: str) -> str:
    """
    Normalize input for SVG filename lookup.

    Examples:
        Jewel   -> jewel
        this is -> this_is
        some of -> some_of
    """

    text = text.strip().lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^a-z'\-\s]", "", text)
    text = re.sub(r"\s+", "_", text)

    return text


def normalize_display_text(text: str) -> str:
    """
    Normalize input for display, IPA, and frequency lookup.

    Keeps spaces instead of converting them to underscores.
    """

    text = text.strip().lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^a-z'\-\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text


# ---------------------------------------------------------------------------
# Frequency rating
# ---------------------------------------------------------------------------


def zipf_to_rating(zipf: float) -> int:
    """
    Convert Zipf frequency into a human-friendly 0-100 score.

    This is a study rating, not a strict statistical percentile.
    """

    if zipf <= 0:
        return 0

    min_zipf = 1.0
    max_zipf = 7.0

    clamped = max(min_zipf, min(zipf, max_zipf))

    return round(((clamped - min_zipf) / (max_zipf - min_zipf)) * 100)


def classify_frequency_commonness(zipf: float) -> str:
    """
    Translate Zipf frequency into a human-readable commonness label.
    """

    if zipf >= 7:
        return "Extremely Common"
    elif zipf >= 6:
        return "Very Common"
    elif zipf >= 5:
        return "Common"
    elif zipf >= 4:
        return "Useful"
    elif zipf >= 3:
        return "Uncommon"
    elif zipf > 0:
        return "Rare"

    return "Not Found"


# ---------------------------------------------------------------------------
# Dynamic grammatical classification
# ---------------------------------------------------------------------------

_WORDNET_READY = False


def ensure_wordnet() -> bool:
    """
    Ensure that the NLTK WordNet corpus is available.
    """

    global _WORDNET_READY

    if _WORDNET_READY:
        return True

    try:
        wn.synsets("test")
        _WORDNET_READY = True
        return True

    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

        try:
            wn.synsets("test")
            _WORDNET_READY = True
            return True

        except LookupError:
            return False


def classify_grammar(text: str) -> str:
    """
    Return a grammatical classification for an English word or phrase.

    Strategy:
    - Known function words are classified manually.
    - Single open-class words are classified dynamically with WordNet.
    - Multi-word phrases are labeled as phrases.
    - Unknown words fall back to suffix rules.
    """

    function_words = {
        "the": "Article / Determiner",
        "a": "Article / Determiner",
        "an": "Article / Determiner",
        "this": "Determiner / Demonstrative Pronoun",
        "that": "Determiner / Demonstrative Pronoun",
        "these": "Determiner / Demonstrative Pronoun",
        "those": "Determiner / Demonstrative Pronoun",
        "i": "Pronoun",
        "me": "Pronoun",
        "you": "Pronoun",
        "he": "Pronoun",
        "him": "Pronoun",
        "she": "Pronoun",
        "her": "Pronoun / Possessive Determiner",
        "it": "Pronoun",
        "we": "Pronoun",
        "us": "Pronoun",
        "they": "Pronoun",
        "them": "Pronoun",
        "my": "Possessive Determiner",
        "your": "Possessive Determiner",
        "his": "Possessive Determiner / Pronoun",
        "its": "Possessive Determiner",
        "our": "Possessive Determiner",
        "their": "Possessive Determiner",
        "mine": "Possessive Pronoun",
        "yours": "Possessive Pronoun",
        "hers": "Possessive Pronoun",
        "ours": "Possessive Pronoun",
        "theirs": "Possessive Pronoun",
        "be": "Verb / Auxiliary",
        "am": "Verb / Auxiliary",
        "is": "Verb / Auxiliary",
        "are": "Verb / Auxiliary",
        "was": "Verb / Auxiliary",
        "were": "Verb / Auxiliary",
        "being": "Verb / Auxiliary",
        "been": "Verb / Auxiliary",
        "do": "Verb / Auxiliary",
        "does": "Verb / Auxiliary",
        "did": "Verb / Auxiliary",
        "have": "Verb / Auxiliary",
        "has": "Verb / Auxiliary",
        "had": "Verb / Auxiliary",
        "will": "Modal Auxiliary",
        "would": "Modal Auxiliary",
        "shall": "Modal Auxiliary",
        "should": "Modal Auxiliary",
        "can": "Modal Auxiliary / Verb",
        "could": "Modal Auxiliary",
        "may": "Modal Auxiliary",
        "might": "Modal Auxiliary",
        "must": "Modal Auxiliary",
        "of": "Preposition",
        "to": "Preposition / Infinitive Marker",
        "in": "Preposition",
        "for": "Preposition",
        "on": "Preposition",
        "with": "Preposition",
        "at": "Preposition",
        "by": "Preposition",
        "from": "Preposition",
        "into": "Preposition",
        "over": "Preposition / Adverb",
        "under": "Preposition / Adverb",
        "through": "Preposition / Adverb",
        "between": "Preposition",
        "about": "Preposition / Adverb",
        "and": "Conjunction",
        "but": "Conjunction",
        "or": "Conjunction",
        "nor": "Conjunction",
        "so": "Conjunction / Adverb",
        "yet": "Conjunction / Adverb",
        "because": "Subordinating Conjunction",
        "if": "Subordinating Conjunction",
        "when": "Subordinating Conjunction / Adverb",
        "while": "Subordinating Conjunction / Noun",
        "although": "Subordinating Conjunction",
        "though": "Subordinating Conjunction / Adverb",
        "not": "Adverb / Negator",
        "no": "Determiner / Adverb",
        "yes": "Interjection / Response Word",
    }

    if text in function_words:
        return function_words[text]

    if " " in text:
        return "Phrase"

    if ensure_wordnet():
        pos_names = {
            "n": "Noun",
            "v": "Verb",
            "a": "Adjective",
            "s": "Adjective",
            "r": "Adverb",
        }

        results = []

        for synset in wn.synsets(text):
            value = pos_names.get(synset.pos())

            if value and value not in results:
                results.append(value)

        if results:
            return " / ".join(results)

    suffix_rules = [
        ("ly", "Adverb"),
        ("ness", "Noun"),
        ("ment", "Noun"),
        ("tion", "Noun"),
        ("sion", "Noun"),
        ("ity", "Noun"),
        ("ship", "Noun"),
        ("able", "Adjective"),
        ("ible", "Adjective"),
        ("al", "Adjective / Noun"),
        ("ous", "Adjective"),
        ("ive", "Adjective"),
        ("ful", "Adjective"),
        ("less", "Adjective"),
        ("ing", "Verb Form / Gerund / Participle"),
        ("ed", "Verb Form / Adjective"),
    ]

    for suffix, classification in suffix_rules:
        if text.endswith(suffix) and len(text) > len(suffix) + 1:
            return classification

    return "Unclassified"


# ---------------------------------------------------------------------------
# IPA to Major system conversion
# ---------------------------------------------------------------------------


def ipa_to_major(ipa_text: str) -> str:
    """
    Convert IPA symbols to Major System digits.
    """

    if not ipa_text or ipa_text == "*":
        return ""

    text = ipa_text.lower()

    for mark in ["ˈ", "ˌ", "ː", ".", "/", "[", "]"]:
        text = text.replace(mark, "")

    multi_symbol_map = [
        ("tʃ", "6"),
        ("dʒ", "6"),
        ("ʧ", "6"),
        ("ʤ", "6"),
    ]

    single_symbol_map = {
        "s": "0",
        "z": "0",
        "t": "1",
        "d": "1",
        "θ": "1",
        "ð": "1",
        "n": "2",
        "ŋ": "2",
        "m": "3",
        "r": "4",
        "ɹ": "4",
        "ɾ": "4",
        "ɚ": "4",
        "ɝ": "4",
        "l": "5",
        "ɫ": "5",
        "ʃ": "6",
        "ʒ": "6",
        "k": "7",
        "g": "7",
        "f": "8",
        "v": "8",
        "p": "9",
        "b": "9",
    }

    digits = []
    i = 0

    while i < len(text):
        matched = False

        for symbol, digit in multi_symbol_map:
            if text.startswith(symbol, i):
                digits.append(digit)
                i += len(symbol)
                matched = True
                break

        if matched:
            continue

        ch = text[i]

        if ch in single_symbol_map:
            digits.append(single_symbol_map[ch])

        i += 1

    return "".join(digits)


# ---------------------------------------------------------------------------
# SVG lookup
# ---------------------------------------------------------------------------


def find_svg_filename(word_or_phrase_key: str) -> Optional[str]:
    """
    Find a matching Gregg shorthand SVG file.

    The lookup first checks the expected lowercase filename directly.
    If that fails, it performs a case-insensitive scan of the SVG directory.

    This matters because macOS is often case-insensitive, while Render/Linux
    is case-sensitive.

    Examples:
        aeolian -> aeolian.svg or Aeolian.svg
        this_is -> this_is.svg or This_Is.svg
    """

    if not word_or_phrase_key:
        return None

    expected_name = f"{word_or_phrase_key}.svg"
    candidate = SVG_DIR / expected_name

    try:
        candidate = candidate.resolve()
    except FileNotFoundError:
        candidate = SVG_DIR / expected_name

    if candidate.exists() and candidate.suffix.lower() == ".svg":
        return candidate.name

    expected_lower = expected_name.lower()

    for svg_file in SVG_DIR.glob("*.svg"):
        if svg_file.name.lower() == expected_lower:
            return svg_file.name

    return None


def svg_stem_to_display_text(stem: str) -> str:
    """
    Convert an SVG filename stem into readable display text.

    Examples:
        this_is -> this is
        as_it_is -> as it is
    """

    return stem.replace("_", " ")


_MAJOR_SYSTEM_INDEX: Optional[Dict[str, List[Dict[str, str]]]] = None


_CMU_MAJOR_MAP = {
    "S": "0",
    "Z": "0",
    "T": "1",
    "D": "1",
    "TH": "1",
    "DH": "1",
    "N": "2",
    "NG": "2",
    "M": "3",
    "R": "4",
    "ER": "4",
    "L": "5",
    "SH": "6",
    "ZH": "6",
    "CH": "6",
    "JH": "6",
    "K": "7",
    "G": "7",
    "F": "8",
    "V": "8",
    "P": "9",
    "B": "9",
}


def cmu_phonemes_to_major(phonemes: str) -> str:
    """
    Convert one CMU/ARPABET pronunciation directly to Major System digits.

    Vowels, stress numbers, H, W, and Y do not contribute digits. The mapping
    mirrors the consonant values already used by ipa_to_major().
    """

    digits = []

    for phoneme in phonemes.split():
        symbol = re.sub(r"\d", "", phoneme).upper()
        digit = _CMU_MAJOR_MAP.get(symbol)

        if digit:
            digits.append(digit)

    return "".join(digits)


def build_major_system_index() -> Dict[str, List[Dict[str, str]]]:
    """
    Build and cache Major System values for the available Gregg SVG outlines.

    The CMU pronunciation database is opened exactly once. Each filename stem
    is then resolved through the first CMU pronunciation for each word, which
    closely follows eng_to_ipa's ordinary dictionary-based lookup while
    avoiding thousands of separate SQLite connections.

    Phrase outlines are supported when every whitespace-delimited word in the
    filename has a CMU pronunciation.
    """

    global _MAJOR_SYSTEM_INDEX

    if _MAJOR_SYSTEM_INDEX is not None:
        return _MAJOR_SYSTEM_INDEX

    cmu_db = (
        Path(ipa.__file__).resolve().parent
        / "resources"
        / "CMU_dict.db"
    )

    pronunciations: Dict[str, str] = {}

    connection = sqlite3.connect(cmu_db)

    try:
        cursor = connection.execute(
            "SELECT word, phonemes FROM dictionary ORDER BY rowid"
        )

        for word, phonemes in cursor:
            key = str(word).casefold()
            pronunciations.setdefault(key, str(phonemes))
    finally:
        connection.close()

    index: Dict[str, List[Dict[str, str]]] = {}

    for svg_file in SVG_DIR.glob("*.svg"):
        if svg_file.suffix.lower() != ".svg":
            continue

        display_text = svg_stem_to_display_text(svg_file.stem)
        words = display_text.casefold().split()

        if not words:
            continue

        major_parts = []
        usable = True

        for word in words:
            phonemes = pronunciations.get(word)

            if phonemes is None:
                usable = False
                break

            major_parts.append(cmu_phonemes_to_major(phonemes))

        if not usable:
            continue

        major_value = "".join(major_parts)

        if not major_value:
            continue

        index.setdefault(major_value, []).append(
            {
                "filename": svg_file.name,
                "display": display_text,
                "svg_url": f"/svg/{svg_file.name}",
                "major_system_value": major_value,
            }
        )

    for matches in index.values():
        matches.sort(
            key=lambda item: (
                len(item["display"]),
                item["display"].casefold(),
                item["filename"].casefold(),
            )
        )

    _MAJOR_SYSTEM_INDEX = index
    return _MAJOR_SYSTEM_INDEX


def find_major_matches(digits: str) -> List[Dict[str, str]]:
    """
    Return every Gregg outline whose complete Major System value
    exactly equals the supplied numerical value.

    The Major index is keyed by complete numerical values, so this lookup
    is a direct dictionary access rather than a scan across all entries.
    """

    if re.fullmatch(r"\d+", digits) is None:
        return []

    return list(build_major_system_index().get(digits, []))



def find_segment_svg_matches(segment_key: str) -> List[Dict[str, str]]:
    """
    Return every Gregg SVG whose filename stem contains segment_key.

    Segment mode uses the same normalization as ordinary lookup:
    - Spaces become underscores.
    - Case is ignored.
    - The search is a substring search against the SVG filename stem.

    Examples:
        the     -> the.svg, there.svg, other.svg, as_the.svg, etc.
        this is -> this_is.svg, as_this_is.svg, etc.
    """

    if not segment_key:
        return []

    segment_lower = segment_key.lower()
    matches = []

    for svg_file in SVG_DIR.glob("*.svg"):
        if svg_file.suffix.lower() != ".svg":
            continue

        stem_lower = svg_file.stem.lower()

        if segment_lower not in stem_lower:
            continue

        display_text = svg_stem_to_display_text(svg_file.stem)

        matches.append(
            {
                "filename": svg_file.name,
                "display": display_text,
                "svg_url": f"/svg/{svg_file.name}",
            }
        )

    def sort_key(item: Dict[str, str]) -> Tuple[int, int, int, str]:
        """
        Sort exact matches first, prefix matches second, then everything else.
        Within each group, shorter and alphabetic entries appear first.
        """

        stem = Path(item["filename"]).stem.lower()

        if stem == segment_lower:
            group = 0
        elif stem.startswith(segment_lower):
            group = 1
        else:
            group = 2

        return (group, len(stem), stem.count("_"), stem)

    return sorted(matches, key=sort_key)


def is_acceptable_live_query(raw_text: str) -> bool:
    """
    Validate partial text for live SVG suggestions.

    This is intentionally more permissive than final lookup validation because
    the user may be midway through typing a word or phrase. Final lookup still
    uses is_acceptable_word().
    """

    cleaned = raw_text.strip().lower()

    if not cleaned:
        return False

    return re.fullmatch(r"[a-z'\-\s]+", cleaned) is not None


def find_live_svg_suggestions(
    search_key: str,
    mode: str = "exact",
    limit: int = 24,
) -> List[Dict[str, str]]:
    """
    Return a small outline-only suggestion list for live typing.

    Modes:
    - exact: exact and prefix filename-stem matches only.
    - segment: substring matches anywhere in the filename stem.

    This function deliberately returns only SVG/display fields. It does not
    compute frequency, grammar, IPA, or Major system values.
    """

    if not search_key:
        return []

    search_lower = search_key.lower()
    mode = "segment" if mode == "segment" else "exact"
    matches = []

    for svg_file in SVG_DIR.glob("*.svg"):
        if svg_file.suffix.lower() != ".svg":
            continue

        stem_lower = svg_file.stem.lower()

        if mode == "segment":
            is_match = search_lower in stem_lower
        else:
            is_match = stem_lower == search_lower or stem_lower.startswith(search_lower)

        if not is_match:
            continue

        display_text = svg_stem_to_display_text(svg_file.stem)

        matches.append(
            {
                "filename": svg_file.name,
                "display": display_text,
                "svg_url": f"/svg/{svg_file.name}",
            }
        )

    def sort_key(item: Dict[str, str]) -> Tuple[int, int, int, str]:
        stem = Path(item["filename"]).stem.lower()

        if stem == search_lower:
            group = 0
        elif stem.startswith(search_lower):
            group = 1
        else:
            group = 2

        return (group, len(stem), stem.count("_"), stem)

    return sorted(matches, key=sort_key)[:limit]


# ---------------------------------------------------------------------------
# Web routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """
    Render the main lookup page.
    """

    return render_template("index.html")


@app.route("/lookup")
def lookup():
    """
    API endpoint for exact word or phrase lookup.
    """

    raw_word = request.args.get("word", "")
    display_text = raw_word.strip()

    if not is_acceptable_word(raw_word):
        if display_text:
            return jsonify({"error": f"{display_text} not found"}), 404

        return jsonify({"error": "Input not found"}), 404

    svg_key = normalize_word(raw_word)
    display_text = normalize_display_text(raw_word)

    freq = zipf_frequency(display_text, "en")
    ipa_text = ipa.convert(display_text)

    if ipa_text == "*":
        ipa_text = ""

    svg_filename = find_svg_filename(svg_key)

    result = {
        "word": display_text,
        "frequency_rating": zipf_to_rating(freq),
        "frequency_commonness": classify_frequency_commonness(freq),
        "classification": classify_grammar(display_text),
        "ipa": ipa_text,
        "major_system_value": ipa_to_major(ipa_text),
        "svg_available": svg_filename is not None,
        "svg_url": f"/svg/{svg_filename}" if svg_filename else None,
    }

    return jsonify(result)


@app.route("/segment")
def segment_lookup():
    """
    API endpoint for segment-mode SVG lookup.

    Segment mode returns every SVG file whose filename stem contains the
    normalized search text. It does not try to classify grammar or IPA,
    because the result set may include many unrelated entries.
    """

    raw_word = request.args.get("word", "")
    display_text = raw_word.strip()

    if not is_acceptable_word(raw_word):
        if display_text:
            return jsonify({"error": f"{display_text} not found"}), 404

        return jsonify({"error": "Input not found"}), 404

    segment_key = normalize_word(raw_word)
    segment_text = normalize_display_text(raw_word)
    matches = find_segment_svg_matches(segment_key)

    return jsonify(
        {
            "segment": segment_text,
            "segment_key": segment_key,
            "raw_query": raw_word.strip(),
            "match_count": len(matches),
            "matches": matches,
        }
    )


@app.route("/suggest")
def suggest_lookup():
    """
    API endpoint for live outline suggestions while the user types.

    Ordinary text input preserves the existing EXACT / SEGMENT suggestion
    behavior. When the frontend Major System toggle is enabled and the query
    contains digits only, the endpoint returns every outline whose complete
    Major System value exactly matches the supplied number. Major results use
    the same multi-card display as segment results.
    """

    raw_word = request.args.get("word", "")
    requested_mode = request.args.get("mode", "exact").lower()
    suggestion_mode = "segment" if requested_mode == "segment" else "exact"
    major_enabled = request.args.get("major", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    try:
        limit = int(request.args.get("limit", 24))
    except ValueError:
        limit = 24

    limit = max(1, min(limit, 60))
    raw_query = raw_word.strip()

    # Major System reverse lookup is opt-in and digit-only. It deliberately
    # bypasses the alphabetic live-query validator below.
    if major_enabled and re.fullmatch(r"\d+", raw_query):
        matches = find_major_matches(raw_query)

        return jsonify(
            {
                "query": raw_query,
                "query_key": raw_query,
                "raw_query": raw_query,
                "mode": "major",
                "match_count": len(matches),
                "matches": matches,
            }
        )

    if not is_acceptable_live_query(raw_word):
        return jsonify(
            {
                "query": normalize_display_text(raw_word),
                "query_key": normalize_word(raw_word),
                "raw_query": raw_word.strip(),
                "mode": suggestion_mode,
                "match_count": 0,
                "matches": [],
            }
        )

    query_key = normalize_word(raw_word)
    query_text = normalize_display_text(raw_word)
    matches = find_live_svg_suggestions(
        query_key,
        mode=suggestion_mode,
        limit=limit,
    )

    return jsonify(
        {
            "query": query_text,
            "query_key": query_key,
            "raw_query": raw_word.strip(),
            "mode": suggestion_mode,
            "match_count": len(matches),
            "matches": matches,
        }
    )


@app.route("/svg/<path:filename>")
def serve_svg(filename):
    """
    Serve a Gregg shorthand SVG file.
    """

    safe_name = Path(filename).name

    if not safe_name.endswith(".svg"):
        abort(404)

    return send_from_directory(SVG_DIR, safe_name, mimetype="image/svg+xml")


if __name__ == "__main__":
    print(f"Using SVG directory: {SVG_DIR}")
    app.run(debug=True, host="127.0.0.1", port=5000)
