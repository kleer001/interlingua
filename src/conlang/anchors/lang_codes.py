"""Wikipedia language-name → BCP-47 / ISO-639 mapping.

Only used as a fallback when a cell has no inner `lang="xx"` attribute. For
non-Latin scripts that's rare; the page authors mark them. For Latin-script
languages the bare `<i>foo</i>` carries no language metadata so we map by the
left-column language name.

ISO 639-1 codes preferred where they exist; 639-3 otherwise.
"""

from __future__ import annotations

# Trailing-comma and whitespace artifacts on Wikipedia are normalized away
# before lookup; comma-disambiguated names ("Chinese, Mandarin") are kept.
WIKIPEDIA_LANG_TO_CODE: dict[str, str] = {
    "Afrikaans": "af",
    "Albanian": "sq",
    "Arabic": "ar",
    "Armenian": "hy",
    "Asturian": "ast",
    # "Australian" appears once in the source; ambiguous (English variant vs
    # an Aboriginal language). Left None until verified.
    "Australian": "",
    "Azerbaijani": "az",
    "Basque": "eu",
    "Batak": "btk",
    "Belarusian": "be",
    "Bengali": "bn",
    "Bosnian": "bs",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Cebuano": "ceb",
    "Chinese, Cantonese": "yue",
    "Chinese, Mandarin": "cmn",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "Egyptian": "arz",
    "English": "en",
    "Estonian": "et",
    "Filipino": "fil",
    "Finnish": "fi",
    "French": "fr",
    "Galician": "gl",
    "Georgian": "ka",
    "German": "de",
    "Greek": "el",
    "Greek (ancient)": "grc",
    "Greek (modern)": "el",
    "Gujarati": "gu",
    "Haitian Creole": "ht",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Indonesian": "id",
    "Irish": "ga",
    "Italian": "it",
    "Japanese": "ja",
    "Kannada": "kn",
    "Kazakh": "kk",
    "Korean": "ko",
    "Kyrgyz": "ky",
    "Latgalian": "ltg",
    "Latin": "la",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Macedonian": "mk",
    "Malay": "ms",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Navajo": "nv",
    "Nepali": "ne",
    "Norwegian": "no",
    "Pashtu": "ps",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Serbian": "sr",
    "Sinhalese": "si",
    "Slovak": "sk",
    "Slovene": "sl",
    "Slovenian": "sl",
    "Somali": "so",
    "Spanish": "es",
    "Sundanese": "su",
    "Swedish": "sv",
    "Tagalog": "tl",
    "Tamil": "ta",
    "Telugu": "te",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    # Uropi is a constructed language, no ISO code assigned.
    "Uropi": "",
    "Vietnamese": "vi",
    "Volapük": "vo",
    "Welsh": "cy",
    "Yiddish": "yi",
}


def normalize_language_name(name: str) -> str:
    """Strip trailing commas/whitespace and collapse internal whitespace.

    Wikipedia editors occasionally leave a stray trailing comma or write
    "Chinese,Mandarin" without the space; normalize both.
    """
    s = name.strip().rstrip(",").strip()
    if "," in s:
        head, _, tail = s.partition(",")
        s = f"{head.strip()}, {tail.strip()}"
    return s


def code_for(name: str) -> str | None:
    """Return ISO/BCP-47 code or None if not mapped."""
    norm = normalize_language_name(name)
    code = WIKIPEDIA_LANG_TO_CODE.get(norm)
    if not code:
        return None
    return code
