"""Language-name normalization and ISO-code lookup."""

from conlang.anchors.lang_codes import code_for, normalize_language_name


def test_normalize_strips_trailing_comma():
    assert normalize_language_name("Armenian,") == "Armenian"
    assert normalize_language_name("  Armenian  ") == "Armenian"


def test_normalize_fixes_missing_space_after_comma():
    assert normalize_language_name("Chinese,Mandarin") == "Chinese, Mandarin"
    assert normalize_language_name("Chinese, Mandarin") == "Chinese, Mandarin"


def test_code_for_iso_639_1():
    assert code_for("English") == "en"
    assert code_for("Japanese") == "ja"
    assert code_for("Arabic") == "ar"
    assert code_for("Russian") == "ru"


def test_code_for_iso_639_3_where_no_2():
    assert code_for("Chinese, Mandarin") == "cmn"
    assert code_for("Chinese, Cantonese") == "yue"
    assert code_for("Cebuano") == "ceb"


def test_code_for_via_normalization():
    # Wikipedia editors sometimes leave trailing commas / missing spaces.
    assert code_for("Armenian,") == "hy"
    assert code_for("Chinese,Mandarin") == "cmn"


def test_code_for_unmapped_returns_none():
    assert code_for("Klingon") is None
    # Uropi has no ISO code (constructed lang); mapped to empty in our dict
    # so the lookup returns None.
    assert code_for("Uropi") is None
