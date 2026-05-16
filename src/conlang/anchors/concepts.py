"""Canonical concept inventory for the anchor pool.

Each ConceptDef records:
- slug: stable canonical ID, matches AnchorEntry.concept
- category: one of the 15 categories from anchor-data-plan.md
- english_seeds: English onomatopoeic words used as Wiktionary lookup keys
- aliases: alternate slugs from upstream sources (e.g., Wikipedia column slugs)
- description: human-readable label

The inventory is the union of:
  (1) the 50 Wikipedia-cross-linguistic-onomatopoeias concepts (already
      ingested in Phase 1) — slugs preserved as-is for traceability,
  (2) gap concepts from anchor-data-plan.md targeted for Wiktionary fill.

`aliases` lets the merge step roll Wikipedia "dog_or_wolf_howling" into
"dog_howling" without losing the source label.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConceptDef:
    slug: str
    category: str
    english_seeds: tuple[str, ...]
    aliases: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""


# fmt: off
CONCEPTS: tuple[ConceptDef, ...] = (
    # ─── mammal ─────────────────────────────────────────────────────────
    ConceptDef("dog_barking",       "mammal",       ("woof", "bark", "arf", "bow-wow"),
               description="Dog barking"),
    ConceptDef("dog_howling",       "mammal",       ("awoo", "howl"),
               aliases=("dog_or_wolf_howling",),
               description="Dog or wolf howling"),
    ConceptDef("cat_meowing",       "mammal",       ("meow", "miaow"),
               description="Cat meowing"),
    ConceptDef("cat_purring",       "mammal",       ("purr",),
               description="Cat purring"),
    ConceptDef("cat_hissing",       "mammal",       ("hiss",),
               description="Cat hissing"),
    ConceptDef("cow_mooing",        "mammal",       ("moo",),
               description="Cow mooing"),
    ConceptDef("pig_grunting",      "mammal",       ("oink", "grunt"),
               description="Pig grunting"),
    ConceptDef("pig_squealing",     "mammal",       ("squeal",),
               description="Pig squealing"),
    ConceptDef("sheep_bleating",    "mammal",       ("baa",),
               description="Sheep bleating"),
    ConceptDef("goat_bleating",     "mammal",       ("bleat",),
               description="Goat bleating"),
    ConceptDef("horse_whinnying",   "mammal",       ("neigh", "whinny"),
               description="Horse whinnying"),
    ConceptDef("horse_galloping",   "mammal",       ("clip-clop", "gallop"),
               description="Horse galloping"),
    ConceptDef("horse_trotting",    "mammal",       ("clip-clop",),
               description="Horse trotting"),
    ConceptDef("donkey_braying",    "mammal",       ("hee-haw", "bray"),
               description="Donkey braying"),
    ConceptDef("elephant_trumpeting", "mammal",     ("trumpet",),
               description="Elephant trumpeting"),
    ConceptDef("lion_roaring",      "mammal",       ("roar",),
               aliases=("lion_tiger_roaring",),
               description="Lion or large cat roaring"),
    ConceptDef("monkey_calling",    "mammal",       ("ooh-ooh-aah-aah",),
               aliases=("monkey_chatting",),
               description="Monkey vocalization"),
    ConceptDef("mouse_squeaking",   "mammal",       ("squeak",),
               description="Mouse squeaking"),
    ConceptDef("bear_growling",     "mammal",       ("growl",),
               description="Bear growling"),
    # ─── bird ───────────────────────────────────────────────────────────
    ConceptDef("rooster_crowing",   "bird",         ("cock-a-doodle-doo", "cockadoodledoo"),
               description="Rooster crowing"),
    ConceptDef("hen_clucking",      "bird",         ("cluck",),
               aliases=("chicken_clucking",),
               description="Hen clucking"),
    ConceptDef("duck_quacking",     "bird",         ("quack",),
               aliases=("duck_calling",),
               description="Duck quacking"),
    ConceptDef("goose_honking",     "bird",         ("honk",),
               aliases=("goose_calling",),
               description="Goose honking"),
    ConceptDef("turkey_gobbling",   "bird",         ("gobble",),
               aliases=("turkey_calling",),
               description="Turkey gobbling"),
    ConceptDef("songbird_singing",  "bird",         ("tweet", "chirp"),
               aliases=("bird_singing",),
               description="Songbird singing"),
    ConceptDef("crow_calling",      "bird",         ("caw",),
               description="Crow or raven calling"),
    ConceptDef("owl_hooting",       "bird",         ("hoot",),
               description="Owl hooting"),
    ConceptDef("dove_cooing",       "bird",         ("coo",),
               description="Dove or pigeon cooing"),
    # ─── reptile / amphibian ────────────────────────────────────────────
    ConceptDef("snake_hissing",     "reptile",      ("hiss",),
               description="Snake hissing"),
    ConceptDef("frog_croaking",     "reptile",      ("ribbit", "croak"),
               description="Frog croaking"),
    # ─── insect / arthropod ─────────────────────────────────────────────
    ConceptDef("bee_buzzing",       "insect",       ("buzz", "bzz"),
               description="Bee or wasp buzzing"),
    ConceptDef("cricket_chirping",  "insect",       ("chirp",),
               description="Cricket chirping"),
    ConceptDef("mosquito_buzzing",  "insect",       ("buzz",),
               description="Mosquito buzzing"),
    ConceptDef("fly_buzzing",       "insect",       ("buzz", "bzzz"),
               description="Fly buzzing"),
    ConceptDef("cicada_droning",    "insect",       ("drone",),
               description="Cicada droning"),
    # ─── marine ─────────────────────────────────────────────────────────
    ConceptDef("whale_calling",     "marine",       ("whale",),
               description="Whale calling"),
    ConceptDef("dolphin_clicking",  "marine",       ("click",),
               description="Dolphin clicking"),
    # ─── water sound ────────────────────────────────────────────────────
    ConceptDef("water_dripping",    "water_sound",  ("drip",),
               description="Water dripping"),
    ConceptDef("water_splashing",   "water_sound",  ("splash",),
               description="Water splashing"),
    ConceptDef("water_gurgling",    "water_sound",  ("gurgle",),
               description="Water gurgling"),
    ConceptDef("water_trickling",   "water_sound",  ("trickle",),
               description="Water trickling"),
    ConceptDef("wave_crashing",     "water_sound",  ("crash",),
               description="Wave crashing"),
    ConceptDef("water_pouring",     "water_sound",  ("pour",),
               description="Water pouring"),
    # ─── fire / wind / weather ──────────────────────────────────────────
    ConceptDef("fire_crackling",    "fire_wind_weather", ("crackle",),
               description="Fire crackling"),
    ConceptDef("wind_howling",      "fire_wind_weather", ("howl",),
               description="Wind howling"),
    ConceptDef("wind_whistling",    "fire_wind_weather", ("whistle",),
               description="Wind whistling"),
    ConceptDef("leaves_rustling",   "fire_wind_weather", ("rustle",),
               description="Leaves rustling"),
    ConceptDef("thunder_clap",      "fire_wind_weather", ("boom", "rumble"),
               description="Thunderclap"),
    ConceptDef("rain_pitterpatter", "fire_wind_weather", ("pitter-patter",),
               description="Rain pitter-patter"),
    # ─── hard impact ────────────────────────────────────────────────────
    ConceptDef("bang",              "hard_impact",  ("bang",),
               description="Bang"),
    ConceptDef("crash",             "hard_impact",  ("crash",),
               description="Crash"),
    ConceptDef("thud",              "hard_impact",  ("thud",),
               description="Thud"),
    ConceptDef("smash",             "hard_impact",  ("smash",),
               description="Smash"),
    ConceptDef("slam",              "hard_impact",  ("slam",),
               description="Slam"),
    ConceptDef("knock",             "hard_impact",  ("knock",),
               description="Knock"),
    ConceptDef("snap",              "hard_impact",  ("snap",),
               description="Snap"),
    ConceptDef("crack",             "hard_impact",  ("crack",),
               description="Crack"),
    ConceptDef("dull_strike",       "hard_impact",  ("thud",),
               aliases=("dull_strike",),
               description="Dull strike (generic)"),
    ConceptDef("falling_strike",    "hard_impact",  ("crash",),
               aliases=("falling_strike",),
               description="Falling strike"),
    ConceptDef("sharp_strike",      "hard_impact",  ("snap",),
               aliases=("sharp_strike",),
               description="Sharp strike"),
    # ─── soft impact ────────────────────────────────────────────────────
    ConceptDef("plop",              "soft_impact",  ("plop",),
               description="Plop"),
    ConceptDef("splat",             "soft_impact",  ("splat",),
               description="Splat"),
    ConceptDef("thump",             "soft_impact",  ("thump",),
               description="Thump"),
    ConceptDef("squish",            "soft_impact",  ("squish",),
               description="Squish"),
    ConceptDef("wet_strike",        "soft_impact",  ("splat",),
               aliases=("wet_strike",),
               description="Wet strike"),
    # ─── resonant ───────────────────────────────────────────────────────
    ConceptDef("clang",             "resonant",     ("clang",),
               description="Clang"),
    ConceptDef("gong",              "resonant",     ("gong",),
               description="Gong"),
    ConceptDef("ding",              "resonant",     ("ding",),
               description="Ding"),
    ConceptDef("ring",              "resonant",     ("ring",),
               description="Ring (bell)"),
    # ─── mechanical / electronic ────────────────────────────────────────
    ConceptDef("tick",              "mechanical",   ("tick",),
               description="Clock ticking"),
    ConceptDef("beep",              "mechanical",   ("beep",),
               description="Electronic beep"),
    ConceptDef("buzz_electronic",   "mechanical",   ("buzz",),
               description="Electronic buzz"),
    ConceptDef("whir",              "mechanical",   ("whir",),
               description="Motor whirring"),
    ConceptDef("hum",               "mechanical",   ("hum",),
               description="Hum (mechanical)"),
    ConceptDef("click",             "mechanical",   ("click",),
               description="Click"),
    ConceptDef("ping",              "mechanical",   ("ping",),
               description="Ping (notification)"),
    ConceptDef("vroom",             "mechanical",   ("vroom",),
               description="Engine vroom"),
    # ─── human nonverbal ────────────────────────────────────────────────
    ConceptDef("laughter",          "human_nonverbal", ("haha", "ha"),
               aliases=("laughter",),
               description="Laughter"),
    ConceptDef("crying",            "human_nonverbal", ("waa", "boohoo"),
               aliases=("baby_crying",),
               description="Crying"),
    ConceptDef("sneezing",          "human_nonverbal", ("achoo",),
               aliases=("sneezing",),
               description="Sneezing"),
    ConceptDef("coughing",          "human_nonverbal", ("cough",),
               aliases=("coughing",),
               description="Coughing"),
    ConceptDef("snoring",           "human_nonverbal", ("zzz", "snore"),
               aliases=("snoring",),
               description="Snoring"),
    ConceptDef("hiccup",            "human_nonverbal", ("hiccup",),
               description="Hiccup"),
    ConceptDef("sigh",              "human_nonverbal", ("sigh",),
               description="Sigh"),
    ConceptDef("gasp",              "human_nonverbal", ("gasp",),
               description="Gasp"),
    ConceptDef("yawning",           "human_nonverbal", ("yawn",),
               aliases=("yawning",),
               description="Yawn"),
    ConceptDef("groan",             "human_nonverbal", ("groan", "ugh"),
               description="Groan"),
    ConceptDef("scream",            "human_nonverbal", ("aaah", "scream"),
               aliases=("scream", "shriek"),
               description="Scream"),
    ConceptDef("heart_beating",     "human_nonverbal", ("thump",),
               aliases=("heart_beating",),
               description="Heart beating"),
    ConceptDef("belching",          "human_nonverbal", ("burp",),
               aliases=("belching",),
               description="Belching"),
    ConceptDef("flatulence",        "human_nonverbal", ("fart",),
               aliases=("flatulence",),
               description="Flatulence"),
    ConceptDef("kiss",              "human_nonverbal", ("mwah", "smooch"),
               aliases=("kiss",),
               description="Kiss"),
    ConceptDef("stuttering",        "human_nonverbal", ("uh",),
               aliases=("stuttering",),
               description="Stuttering"),
    # ─── movement ───────────────────────────────────────────────────────
    ConceptDef("whoosh",            "movement",     ("whoosh",),
               description="Whoosh (air/wind)"),
    ConceptDef("zoom",              "movement",     ("zoom",),
               description="Zoom (fast motion)"),
    ConceptDef("swish",             "movement",     ("swish",),
               description="Swish (fabric)"),
    ConceptDef("flap",              "movement",     ("flap",),
               description="Flapping (wings, cloth)"),
    # ─── texture / eating ───────────────────────────────────────────────
    ConceptDef("crunch",            "texture_eating", ("crunch",),
               description="Crunch"),
    ConceptDef("slurp",             "texture_eating", ("slurp",),
               description="Slurp"),
    ConceptDef("smack_eat",         "texture_eating", ("smack",),
               description="Smack (lips while eating)"),
    ConceptDef("gulp",              "texture_eating", ("gulp",),
               description="Gulp"),
    ConceptDef("scratch",           "texture_eating", ("scratch",),
               description="Scratch"),
    ConceptDef("scrape",            "texture_eating", ("scrape",),
               description="Scrape"),
    ConceptDef("biting",            "texture_eating", ("chomp",),
               aliases=("biting",),
               description="Biting"),
    ConceptDef("eating",            "texture_eating", ("nom", "munch"),
               aliases=("eating_food",),
               description="Eating food"),
    ConceptDef("drinking",          "texture_eating", ("glug",),
               aliases=("drinking",),
               description="Drinking"),
    ConceptDef("swallowing",        "texture_eating", ("gulp",),
               aliases=("swallowing",),
               description="Swallowing"),
    # ─── exclamation / affect ───────────────────────────────────────────
    ConceptDef("ouch",              "exclamation_affect", ("ouch", "ow"),
               description="Ouch"),
    ConceptDef("wow",               "exclamation_affect", ("wow",),
               description="Wow (amazement)"),
    ConceptDef("ugh",               "exclamation_affect", ("ugh",),
               description="Ugh (disgust)"),
    ConceptDef("eww",               "exclamation_affect", ("eww", "yuck"),
               description="Eww (revulsion)"),
    ConceptDef("hmm",               "exclamation_affect", ("hmm",),
               aliases=("pausing_or_thinking",),
               description="Hmm (thinking)"),
    ConceptDef("hushing",           "exclamation_affect", ("shh",),
               aliases=("hushing",),
               description="Hushing"),
    ConceptDef("brushing_teeth",    "texture_eating", ("brush",),
               aliases=("brushing_teeth",),
               description="Brushing teeth"),
)
# fmt: on


# Build lookup tables.
BY_SLUG: dict[str, ConceptDef] = {c.slug: c for c in CONCEPTS}
ALIAS_TO_SLUG: dict[str, str] = {a: c.slug for c in CONCEPTS for a in (c.aliases + (c.slug,))}


def canonical_slug(label_or_slug: str) -> str | None:
    """Map a slug (Wikipedia or canonical) to its canonical inventory slug.

    Returns None if the label doesn't match anything in the inventory.
    """
    return ALIAS_TO_SLUG.get(label_or_slug)


def concepts_by_category() -> dict[str, list[ConceptDef]]:
    out: dict[str, list[ConceptDef]] = {}
    for c in CONCEPTS:
        out.setdefault(c.category, []).append(c)
    return out
