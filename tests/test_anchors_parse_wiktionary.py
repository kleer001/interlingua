"""Parser tests on synthetic Wiktionary HTML."""

from conlang.lab.concepts import ConceptDef
from conlang.anchors.parse_wiktionary import parse_seed_html

_DOG_BARK_PAGE = """
<h2>English</h2>
<h3>Etymology 2</h3>
<h4>Interjection</h4>
<p>Definition stuff.</p>
<h5>Translations</h5>
<div class="NavFrame">
  <div class="NavHead">the sound of a dog barking</div>
  <table class="translations">
    <tr><td><ul>
      <li>Arabic: <span class="Arab" lang="ar">هَوْ</span>
          <span class="mention-gloss-paren annotation-paren">(</span>
          <span class="tr Latn" lang="ar-Latn">haw</span>
          <span class="mention-gloss-paren annotation-paren">)</span></li>
      <li>Chinese:
        <dl><dd>Mandarin:
          <span class="Hani" lang="cmn">汪</span>
          <span class="mention-gloss-paren annotation-paren">(</span>
          <span class="tr Latn" lang="cmn-Latn">wāng</span>
          <span class="mention-gloss-paren annotation-paren">)</span>
        </dd></dl>
      </li>
      <li>Japanese:
        <span class="Jpan" lang="ja">ワンワン</span>
        <span class="mention-gloss-paren annotation-paren">(</span>
        <span class="tr">wanwan</span>
        <span class="mention-gloss-paren annotation-paren">)</span></li>
      <li>Russian:
        <span class="Cyrl" lang="ru">гав</span>
        <span class="mention-gloss-paren annotation-paren">(</span>
        <span class="tr Latn" lang="ru-Latn">gav</span>
        <span class="mention-gloss-paren annotation-paren">)</span></li>
      <li>Finnish: <span class="Latn" lang="fi">hau</span></li>
    </ul></td></tr>
  </table>
</div>
<h4>Verb</h4>
<h5>Translations</h5>
<div class="NavFrame">
  <div class="NavHead">of a dog: to bark</div>
  <table class="translations">
    <tr><td><ul>
      <li>Russian: <span class="Cyrl" lang="ru">ла́ять</span>
          <span class="mention-gloss-paren annotation-paren">(</span>
          <span class="tr Latn" lang="ru-Latn">lájatʹ</span>
          <span class="mention-gloss-paren annotation-paren">)</span></li>
    </ul></td></tr>
  </table>
</div>
"""


def _woof_concept() -> ConceptDef:
    return ConceptDef(
        slug="dog_barking",
        category="mammal",
        english_seeds=("woof",),
        aliases=(),
        description="Dog barking",
    )


def _parse(html: str, seed: str = "woof"):
    return parse_seed_html(
        html,
        seed=seed,
        source_url=f"https://en.wiktionary.org/wiki/{seed}",
        source_revid="rev1",
        captured_at="2026-05-16",
        concepts=[_woof_concept()],
    )


def test_extracts_native_plus_romanization():
    entries = _parse(_DOG_BARK_PAGE)
    ja = [e for e in entries if e.language_code == "ja"]
    assert len(ja) == 1
    assert ja[0].orthography == "ワンワン"
    assert ja[0].romanization == "wanwan"


def test_handles_nested_chinese_sub_language():
    entries = _parse(_DOG_BARK_PAGE)
    cmn = [e for e in entries if e.language_code == "cmn"]
    assert len(cmn) == 1
    assert cmn[0].orthography == "汪"
    assert cmn[0].romanization == "wāng"
    assert "Mandarin" in cmn[0].language  # nested language label is preserved


def test_handles_arabic_native_plus_romanization():
    entries = _parse(_DOG_BARK_PAGE)
    ar = [e for e in entries if e.language_code == "ar"]
    assert len(ar) == 1
    assert ar[0].orthography == "هَوْ"
    assert ar[0].romanization == "haw"


def test_handles_latin_script_only_language():
    entries = _parse(_DOG_BARK_PAGE)
    fi = [e for e in entries if e.language_code == "fi"]
    assert len(fi) == 1
    assert fi[0].orthography == "hau"
    assert fi[0].romanization is None


def test_drops_verb_section_translations():
    """Russian *ла́ять* (to bark) is under a Verb header — must NOT be emitted
    for the onomatopoeic dog_barking concept."""
    entries = _parse(_DOG_BARK_PAGE)
    forms = {e.orthography for e in entries if e.language_code == "ru"}
    assert "гав" in forms
    assert "ла́ять" not in forms


def test_only_emits_for_concepts_matching_gloss():
    """If we give the parser a concept whose keywords don't match the
    NavHead gloss, no rows should come out."""
    # A concept that explicitly doesn't share keywords with "dog barking"
    unrelated = ConceptDef(
        slug="bee_buzzing",
        category="insect",
        english_seeds=("woof",),  # forces it to be considered for the woof page
        aliases=(),
        description="Bee or wasp buzzing",
    )
    entries = parse_seed_html(
        _DOG_BARK_PAGE,
        seed="woof",
        source_url="https://en.wiktionary.org/wiki/woof",
        source_revid="rev1",
        captured_at="2026-05-16",
        concepts=[unrelated],
    )
    # NavHead "the sound of a dog barking" doesn't match {bee, wasp, insect, buzzing}
    assert entries == []
