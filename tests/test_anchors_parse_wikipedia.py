"""Parser tests on synthetic HTML fixtures mirroring real Wikipedia patterns."""

from bs4 import BeautifulSoup

from conlang.anchors.parse_wikipedia import parse_cell, parse_html, slugify_concept


def _cell(html: str):
    return BeautifulSoup(f"<table><tr><td>{html}</td></tr></table>", "lxml").td


def test_slugify_concept_handles_punctuation():
    assert slugify_concept("Cat meowing") == "cat_meowing"
    assert slugify_concept("Dog or wolf howling") == "dog_or_wolf_howling"
    assert slugify_concept("  Pig grunting!  ") == "pig_grunting"
    assert slugify_concept("Lion/tiger roaring") == "lion_tiger_roaring"


def test_parse_cell_latin_script_with_alternatives():
    """English-style cell: bare <i> tags separated by commas."""
    c = _cell("<i>woof</i>, <i>arf</i>, <i>bow wow</i>")
    out = parse_cell(c, fallback_language_code="en")
    assert [p.orthography for p in out] == ["woof", "arf", "bow wow"]
    for p in out:
        assert p.romanization is None
        assert p.language_code == "en"


def test_parse_cell_native_plus_romanization_span():
    """Japanese-style: lang-text span + lang-romanization span in parens."""
    c = _cell(
        '<span title="Japanese-language text"><i lang="ja">ワンワン</i></span>'
        ' (<span title="Japanese-language romanization">'
        '<i lang="ja-Latn">wan wan</i></span>)'
    )
    out = parse_cell(c)
    assert len(out) == 1
    p = out[0]
    assert p.orthography == "ワンワン"
    assert p.romanization == "wan wan"
    assert p.language_code == "ja"
    assert p.ipa is None


def test_parse_cell_multiple_native_pairs_without_comma():
    """Arabic-style: two native+romanization pairs separated by space, no comma."""
    c = _cell(
        '<span title="Arabic-language text"><span dir="rtl" lang="ar">مُواَء</span></span>'
        ' (<span title="Arabic-language romanization"><i lang="ar-Latn">muwa</i></span>)'
        ' <span title="Arabic-language text"><span dir="rtl" lang="ar">مياو</span></span>'
        ' (<span title="Arabic-language romanization"><i lang="ar-Latn">miao</i></span>)'
    )
    out = parse_cell(c)
    assert len(out) == 2
    assert out[0].orthography == "مُواَء"
    assert out[0].romanization == "muwa"
    assert out[1].orthography == "مياو"
    assert out[1].romanization == "miao"


def test_parse_cell_native_then_bare_italic_in_parens():
    """Chinese-style: NATIVE (<i>roman</i>) with no lang-text wrapper."""
    c = _cell("哞 (<i>mōu</i>)")
    out = parse_cell(c, fallback_language_code="cmn")
    assert len(out) == 1
    assert out[0].orthography == "哞"
    assert out[0].romanization == "mōu"
    assert out[0].language_code == "cmn"


def test_parse_cell_with_ipa_span():
    """Cell with an IPA span attached to a native form."""
    c = _cell(
        '<span title="Czech-language text"><i lang="cs">chrup</i></span>'
        ' <span class="IPA" lang="cs-Latn-fonipa">[xr̝up]</span>'
    )
    out = parse_cell(c)
    assert len(out) == 1
    assert out[0].orthography == "chrup"
    assert out[0].ipa == "[xr̝up]"


def test_parse_cell_disambiguator_into_notes():
    """English '(small dog)' next to a form should become a note, not be lost."""
    c = _cell("<i>yip yip</i> (small dog)")
    out = parse_cell(c, fallback_language_code="en")
    assert len(out) == 1
    assert out[0].orthography == "yip yip"
    assert out[0].finalize_notes() == "small dog"


def test_parse_cell_strips_citation_sup():
    """<sup class="reference"> citations should not pollute orthography or notes."""
    c = _cell('<i>woof</i><sup class="reference">[1]</sup>')
    out = parse_cell(c, fallback_language_code="en")
    assert len(out) == 1
    assert out[0].orthography == "woof"
    assert out[0].finalize_notes() is None


def test_parse_html_minimal_table():
    """End-to-end on a tiny synthetic page."""
    html = """
    <h2>Animal sounds</h2>
    <h3>Mammal sounds</h3>
    <h4>Cats and dogs</h4>
    <table class="wikitable">
      <tr><th>Language</th><th>Cat meowing</th><th>Dog barking</th></tr>
      <tr><th>English</th>
          <td><i>meow</i></td>
          <td><i>woof</i>, <i>arf</i></td></tr>
      <tr><th>Japanese</th>
          <td><span title="Japanese-language text"><i lang="ja">ニャー</i></span>
              (<span title="Japanese-language romanization"><i lang="ja-Latn">nyā</i></span>)</td>
          <td><span title="Japanese-language text"><i lang="ja">ワンワン</i></span>
              (<span title="Japanese-language romanization">
                <i lang="ja-Latn">wan wan</i></span>)</td></tr>
    </table>
    """
    entries = parse_html(
        html,
        source="test",
        source_url="https://example/",
        source_revid="0",
        captured_at="2026-05-16",
    )
    # 1 English meow + 2 English barks + 1 Japanese meow + 1 Japanese bark = 5
    assert len(entries) == 5
    concepts = sorted(set(e.concept for e in entries))
    assert concepts == ["cat_meowing", "dog_barking"]
    cats = sorted(set(e.category for e in entries))
    assert cats == ["Cats and dogs"]
    section_paths = sorted(set(e.extra["section_path"] for e in entries))
    assert section_paths == ["Animal sounds / Mammal sounds / Cats and dogs"]
    # Japanese rows have both orthography and romanization
    ja_bark = [e for e in entries if e.language == "Japanese" and e.concept == "dog_barking"]
    assert len(ja_bark) == 1
    assert ja_bark[0].orthography == "ワンワン"
    assert ja_bark[0].romanization == "wan wan"
    assert ja_bark[0].language_code == "ja"
