"""Phase-6 skeleton: bind concepts to semantic-attribute bundles.

In the anchor-pool design (anchor-pool-sketch.md §"Attribute bucket"), each
animal concept is not a single point in concept space but a *cloud* of
attribute points. The attribute list for snake is its real anchor —
{close-to-ground, elongated, sudden-strike, ...} — and the snake-shaped
cloud is what the LLM-extracted concepts get interpolated against.

The registry below is the *starter* hand-curated hypothesis (per the plan's
Phase 6 step). The real attribute set will come from the SAE feature
clustering once Stage 4 stabilizes; this file is the wiring that lets
Phase 6 emit `(concept, attribute, phonological_signature)` rows the moment
that embedding is available.

Cultural attributes are flagged separately because they're not universal
and may need per-language weighting in the final Voronoi tessellation.

This module does NOT do the embedding step itself. The function
`build_attribute_anchor_table()` returns rows with a stub
`attribute_embedding=None`; Stage 5 will fill that field by passing the
attribute text through Gemma 2 2B + the slice's SAE encoder and reading
back the active-feature vector.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .schema import AnchorEntry, read_jsonl


@dataclass(frozen=True)
class AttributeBundle:
    concept: str
    attributes: tuple[str, ...]
    cultural_attributes: tuple[str, ...] = field(default_factory=tuple)


# fmt: off
ATTRIBUTE_REGISTRY: dict[str, AttributeBundle] = {
    "snake_hissing": AttributeBundle(
        concept="snake_hissing",
        attributes=(
            "close-to-ground", "elongated", "smooth-scaled", "flexible",
            "sinuous", "sudden-strike", "concealed", "cold-blooded",
            "venomous", "silent-movement", "fork-tongued", "shedding-skin",
            "dread-inducing", "primal-fear-trigger", "alarm-signal",
        ),
        cultural_attributes=(
            "wisdom-symbol-asian",
            "evil-bringer-abrahamic",
            "fertility-renewal-mesoamerican",
            "cunning-trickster-mediterranean",
            "healing-caduceus-greek",
            "ritual-taboo-judaic",
            "kundalini-life-force-hindu",
            "primordial-chaos-jormungandr-norse",
            "fortune-zodiac-chinese",
            "ancestor-spirit-west-african",
        ),
    ),
    "dog_barking": AttributeBundle(
        concept="dog_barking",
        attributes=(
            "domestic", "loyal", "alert", "protective", "playful",
            "fast-runner", "carnivore", "four-legged", "tail-wagging",
            "scent-tracking", "pack-bonded",
            "comforting-presence", "warning-signal", "alarm-trigger",
        ),
        cultural_attributes=(
            "faithful-companion-western",
            "ritually-unclean-islamic",
            "ritually-impure-japanese-folkloric",
            "psychopomp-anubis-egyptian",
            "demonic-mesopotamian",
            "hunting-noble-medieval-european",
            "fortune-zodiac-chinese",
            "loyalty-virtue-confucian",
            "scavenger-pariah-south-asian",
            "spirit-guide-mesoamerican-xolotl",
        ),
    ),
    "cat_meowing": AttributeBundle(
        concept="cat_meowing",
        attributes=(
            "domestic", "independent", "agile", "stealthy", "small-predator",
            "nocturnal", "feline", "four-legged", "retractable-claws",
            "silent-stalking", "whiskered", "purring-capable",
            "comforting-purr", "mysterious-presence", "contemplative-mood",
        ),
        cultural_attributes=(
            "sacred-bastet-egyptian",
            "maneki-neko-fortune-japanese",
            "unlucky-black-cat-european",
            "witch-familiar-medieval-european",
            "aloof-virtue-western",
            "nine-lives-superstition-pan-cultural",
            "bakeneko-demon-japanese",
            "prosperity-spirit-east-asian",
            "evil-eye-protector-mediterranean",
            "mischievous-trickster-folkloric",
        ),
    ),
    "cow_mooing": AttributeBundle(
        concept="cow_mooing",
        attributes=(
            "large", "domesticated-livestock", "herbivore", "ruminant",
            "slow-moving", "milk-producing", "four-legged", "horned",
            "gentle-natured", "herd-animal", "dewlap", "cloven-hoofed",
            "pastoral-calm", "mournful-low-call", "abundance-presence",
        ),
        cultural_attributes=(
            "sacred-gau-mata-hindu",
            "maternal-symbol-pan-cultural",
            "sacrificial-victim-abrahamic",
            "golden-calf-idolatry-judaic",
            "hathor-goddess-egyptian",
            "bull-virility-mediterranean",
            "audhumla-primordial-cow-norse",
            "dharma-protector-buddhist",
            "agricultural-foundation-near-eastern",
            "prosperity-pastoral-celtic",
        ),
    ),
    "pig_grunting": AttributeBundle(
        concept="pig_grunting",
        attributes=(
            "domestic-livestock", "omnivore", "intelligent", "muddy", "stout",
            "four-legged", "rooting", "snorting", "curly-tailed",
            "fast-fattening", "vocal", "social",
            "contented-grunting", "comedic-figure", "lazy-presence",
        ),
        cultural_attributes=(
            "ritually-unclean-islamic",
            "ritually-unclean-judaic",
            "gluttony-vice-christian",
            "year-of-pig-fortune-chinese",
            "piggy-bank-luck-folk-western",
            "demeter-fertility-sacrifice-greek",
            "kalydonian-boar-myth-greek",
            "spirit-pig-polynesian",
            "boar-warrior-celtic",
            "taboo-meat-many-traditions",
        ),
    ),
    "horse_whinnying": AttributeBundle(
        concept="horse_whinnying",
        attributes=(
            "large", "fast-runner", "herbivore", "domesticated", "four-legged",
            "hooved", "strong", "mane-tailed", "grass-grazer", "herd-animal",
            "war-mount", "tall-withered",
            "noble-presence", "eager-anticipation", "restless-spirit",
        ),
        cultural_attributes=(
            "freedom-symbol-western",
            "warrior-mount-medieval-european",
            "trojan-deception-greek",
            "pegasus-flying-greek",
            "sleipnir-odin-norse",
            "year-of-horse-zodiac-chinese",
            "hippomania-celtic",
            "ancestor-spirit-mongol",
            "ashvamedha-sacrifice-vedic",
            "horseman-apocalypse-christian",
        ),
    ),
    "rooster_crowing": AttributeBundle(
        concept="rooster_crowing",
        attributes=(
            "diurnal", "domestic-bird", "dawn-call", "loud", "territorial",
            "two-legged", "feathered", "combed-crested", "spurred",
            "masculine-display", "ground-pecking", "brilliant-plumage",
            "dawn-arousal", "vigilant-presence", "masculine-pride",
        ),
        cultural_attributes=(
            "vigilance-virtue-christian",
            "denial-peter-biblical",
            "year-of-rooster-zodiac-chinese",
            "weathervane-protective-european",
            "dispelling-darkness-persian-avestan",
            "fire-bird-mythic-pan-cultural",
            "gallic-national-symbol-french",
            "demon-banishment-folk-european",
            "yang-energy-chinese",
            "masculine-archetype-pan-cultural",
        ),
    ),
    "owl_hooting": AttributeBundle(
        concept="owl_hooting",
        attributes=(
            "nocturnal", "predator", "silent-flight", "large-eyes",
            "two-legged", "feathered", "perched", "head-rotating", "taloned",
            "camouflaged", "hooting-resonant", "solitary",
            "eerie-presence", "watchful-vigilance", "death-foreboding",
        ),
        cultural_attributes=(
            "wisdom-athena-greek",
            "ill-omen-death-pan-cultural",
            "witch-familiar-medieval-european",
            "lakshmi-vahana-hindu",
            "sorcery-medieval-european",
            "soul-of-departed-mexican-folkloric",
            "night-terror-slavic",
            "blodeuwedd-betrayal-welsh",
            "ancestor-spirit-indigenous-north-american",
            "knowledge-keeper-pan-cultural",
        ),
    ),
    "bee_buzzing": AttributeBundle(
        concept="bee_buzzing",
        attributes=(
            "small", "flying", "stinging", "colony", "busy", "honey-producing",
            "six-legged", "winged", "hexagonal-comb", "pollinator",
            "dancer-communicator", "golden-yellow",
            "industrious-presence", "soothing-hum", "sting-alarm",
        ),
        cultural_attributes=(
            "industriousness-virtue-western",
            "sweetness-honey-pan-cultural",
            "telling-the-bees-celtic-folk",
            "royal-power-napoleonic-french",
            "melissae-priestesses-greek",
            "divine-messenger-egyptian",
            "bee-shaman-indigenous-pan-cultural",
            "paradise-honey-quranic-islamic",
            "ah-muzen-cab-bee-god-mayan",
            "hive-mind-collective-modern-western",
        ),
    ),
    "frog_croaking": AttributeBundle(
        concept="frog_croaking",
        attributes=(
            "amphibian", "wet", "jumping", "small", "cold-blooded",
            "rain-associated", "four-legged", "smooth-skinned", "bulging-eyed",
            "metamorphic", "water-born", "croak-resonant",
            "nocturnal-chorus", "slimy-aversion", "rain-anticipation",
        ),
        cultural_attributes=(
            "transformation-metamorphosis-pan-cultural",
            "luck-fertility-east-asian",
            "plague-egyptian-biblical",
            "rain-bringer-shamanic-mesoamerican",
            "witch-familiar-medieval-european",
            "prince-enchantment-fairy-tale-european",
            "heqet-birth-goddess-egyptian",
            "jin-chan-three-legged-coin-toad-chinese",
            "ritual-taboo-leviticus-judaic",
            "soul-symbol-mesoamerican",
        ),
    ),
    "lion_roaring": AttributeBundle(
        concept="lion_roaring",
        attributes=(
            "large-predator", "mane", "loud", "regal", "savannah",
            "four-legged", "feline", "carnivore", "ambush-hunter",
            "pride-social", "tawny", "muscular",
            "regal-presence", "terror-trigger", "dominance-aura",
        ),
        cultural_attributes=(
            "royalty-symbol-western",
            "judah-tribe-biblical",
            "aslan-christ-allegory-modern-christian",
            "narasimha-vishnu-hindu",
            "dharma-protector-buddhist",
            "lion-gate-mycenaean",
            "sphinx-riddler-egyptian-greek",
            "nemean-lion-herculean-greek",
            "chinese-guardian-lion-buddhist",
            "courage-virtue-pan-cultural",
        ),
    ),
    "mouse_squeaking": AttributeBundle(
        concept="mouse_squeaking",
        attributes=(
            "tiny", "fast", "fearful", "nibbling", "four-legged", "tailed",
            "rodent", "scurrying", "twitching-whiskered", "crepuscular",
            "granary-pest", "fecund-breeder",
            "timid-presence", "startling-skitter", "vulnerability-evoking",
        ),
        cultural_attributes=(
            "ganesha-vahana-hindu",
            "apollo-smintheus-greek",
            "year-of-rat-zodiac-chinese",
            "household-pest-pan-cultural",
            "plague-bringer-medieval-european",
            "amulet-protective-egyptian",
            "shapeshifter-witch-form-folkloric",
            "ritual-taboo-leviticus-judaic",
            "humble-meekness-virtue-christian",
            "trickster-creator-indigenous-north-american",
        ),
    ),
    "crow_calling": AttributeBundle(
        concept="crow_calling",
        attributes=(
            "black", "intelligent", "scavenger", "loud", "perched",
            "two-legged", "feathered", "tool-using", "social-flock",
            "omnivore", "raucous", "oily-plumage",
            "ominous-presence", "foreboding-call", "eerie-watchfulness",
        ),
        cultural_attributes=(
            "hugin-munin-odin-norse",
            "apollo-messenger-greek",
            "yatagarasu-three-legged-japanese",
            "war-bird-morrigan-celtic",
            "noah-flood-biblical",
            "ancestor-spirit-pacific-northwest-indigenous",
            "trickster-creator-pan-indigenous",
            "soul-carrier-mexican-folkloric",
            "ill-omen-western",
            "judgment-day-eye-pecker-folkloric",
        ),
    ),
    "wolf_howling": AttributeBundle(
        concept="dog_howling",  # canonical slug includes wolf
        attributes=(
            "wild", "predator", "pack", "loud", "long-howl", "four-legged",
            "carnivore", "gray-furred", "yellow-eyed", "scent-marking",
            "dawn-dusk-active", "lone-or-packed",
            "lonely-presence", "dread-inducing", "primal-wilderness-call",
        ),
        cultural_attributes=(
            "werewolf-lycanthropy-european",
            "demonic-medieval-christian",
            "romulus-remus-she-wolf-roman",
            "fenrir-doom-norse",
            "ancestor-spirit-mongol-borte-chino",
            "spirit-guide-pacific-northwest-indigenous",
            "hunter-virtue-many-indigenous",
            "raksha-protector-buddhist",
            "death-omen-folkloric-pan-cultural",
            "fenris-binding-mythic-norse",
        ),
    ),
    # ─── Batch 1: other mammals ─────────────────────────────────────────
    "cat_hissing": AttributeBundle(
        concept="cat_hissing",
        attributes=(
            "feline", "four-legged", "arched-back", "bared-teeth", "raised-fur",
            "ears-flattened", "tail-puffed", "aggressive-stance",
            "defensive-posture", "sibilant-exhalation", "breathy", "snake-mimic",
            "threat-signal", "aggression-display", "fear-amplifier",
        ),
        cultural_attributes=(
            "feline-aggression-pan-cultural",
            "snake-mimicry-defense-naturalist",
            "witch-cat-medieval-european",
            "bakeneko-warning-japanese",
            "jealous-rivalry-folkloric-european",
            "taboo-rage-buddhist",
            "evil-spirit-warding-mediterranean",
            "hostility-omen-celtic",
            "demon-form-japanese-folkloric",
            "predator-warning-pan-cultural",
        ),
    ),
    "cat_purring": AttributeBundle(
        concept="cat_purring",
        attributes=(
            "feline", "four-legged", "vibrating-throat", "low-frequency",
            "sustained", "breath-rhythm", "kneading-paws", "eyes-narrowed",
            "relaxed-posture", "lap-bound", "soft-rumble",
            "healing-frequency-claimed",
            "comforting-presence", "contentment-signal", "soothing-mood",
        ),
        cultural_attributes=(
            "domestic-bliss-western",
            "healing-vibration-modern-belief",
            "mother-cat-nurturing-pan-cultural",
            "prosperity-omen-east-asian",
            "witch-magic-medieval-european",
            "sacred-purr-bastet-egyptian",
            "lap-cat-aristocratic-european",
            "contentment-virtue-buddhist",
            "household-blessing-folk-european",
            "nine-lives-aura-superstition",
        ),
    ),
    "donkey_braying": AttributeBundle(
        concept="donkey_braying",
        attributes=(
            "long-eared", "sturdy", "four-legged", "hooved", "herbivore",
            "mane-tailed", "slow-walking", "stubborn-set", "loud-bray",
            "harsh-call", "gray-coated", "pack-animal",
            "comedic-figure", "weary-presence", "stubborn-aura",
        ),
        cultural_attributes=(
            "balaam-prophet-biblical",
            "christ-mount-palm-sunday-christian",
            "foolishness-archetype-western",
            "midas-ears-greek",
            "set-storm-god-egyptian",
            "humility-virtue-christian",
            "stubborn-mule-anglo-saxon",
            "peasant-beast-medieval-european",
            "jenny-mother-pastoral-celtic",
            "democratic-party-symbol-american",
        ),
    ),
    "elephant_trumpeting": AttributeBundle(
        concept="elephant_trumpeting",
        attributes=(
            "massive", "four-legged", "trunked", "tusked", "gray-skinned",
            "large-eared", "slow-moving", "herbivore", "intelligent",
            "social-herd", "long-memory", "columnar-legs",
            "regal-presence", "awe-evoking", "primal-power",
        ),
        cultural_attributes=(
            "ganesha-deity-hindu",
            "dharma-bearer-buddhist",
            "royal-mount-southeast-asian",
            "ivory-luxury-pan-cultural",
            "white-elephant-omen-thai-burmese",
            "memory-virtue-pan-cultural",
            "hannibal-war-elephant-roman",
            "ancestor-spirit-african",
            "airavata-indra-mount-hindu",
            "wisdom-elder-african",
        ),
    ),
    "goat_bleating": AttributeBundle(
        concept="goat_bleating",
        attributes=(
            "horned", "four-legged", "cloven-hoofed", "milk-producing",
            "agile-climber", "herd-animal", "herbivore", "bearded",
            "rectangular-pupil", "sure-footed", "mountain-dwelling", "hardy",
            "pastoral-presence", "mischievous-aura", "scapegoat-burden",
        ),
        cultural_attributes=(
            "scapegoat-yom-kippur-judaic",
            "capricorn-zodiac-pan-cultural",
            "satyr-pan-greek",
            "devil-baphomet-occult-medieval",
            "mendes-deity-egyptian",
            "sacrificial-victim-abrahamic",
            "yule-goat-norse-scandinavian",
            "year-of-goat-zodiac-chinese",
            "pastoral-foundation-mediterranean",
            "milk-providers-near-eastern",
        ),
    ),
    "monkey_calling": AttributeBundle(
        concept="monkey_calling",
        attributes=(
            "primate", "four-handed", "agile", "tree-dwelling", "social-troop",
            "intelligent", "vocal", "expressive-faced", "grasping", "omnivore",
            "tailed", "climber",
            "mischievous-presence", "comedic-figure", "anthropomorphic-mirror",
        ),
        cultural_attributes=(
            "hanuman-deity-hindu",
            "sun-wukong-trickster-chinese",
            "three-wise-monkeys-buddhist",
            "evolution-ancestor-modern-western",
            "year-of-monkey-zodiac-chinese",
            "howler-monkey-god-mesoamerican-mayan",
            "taboo-meat-some-african",
            "foolishness-emulation-western",
            "jungle-spirit-southeast-asian",
            "mischief-archetype-pan-cultural",
        ),
    ),
    "pig_squealing": AttributeBundle(
        concept="pig_squealing",
        attributes=(
            "domestic-livestock", "omnivore", "intelligent", "muddy", "stout",
            "four-legged", "high-pitched", "piercing", "sudden-burst",
            "distress-call", "vocal", "fast-fattening",
            "distress-trigger", "alarm-signal", "terror-evoking",
        ),
        cultural_attributes=(
            "slaughter-association-pan-cultural",
            "witch-shrieking-medieval-european",
            "banshee-comparison-celtic",
            "distress-omen-folk-european",
            "demeter-sacrifice-greek",
            "ritually-unclean-islamic",
            "ritually-unclean-judaic",
            "boar-hunt-prize-celtic",
            "comedic-overreaction-western",
            "year-of-pig-zodiac-chinese",
        ),
    ),
    "sheep_bleating": AttributeBundle(
        concept="sheep_bleating",
        attributes=(
            "woolly", "four-legged", "cloven-hoofed", "herd-animal", "herbivore",
            "gentle-natured", "ruminant", "small-horned", "mild-eyed",
            "fleece-producing", "pastoral", "follower",
            "docile-presence", "pastoral-calm", "vulnerability-evoking",
        ),
        cultural_attributes=(
            "lamb-of-god-christian",
            "sacrificial-victim-abrahamic",
            "golden-fleece-jason-greek",
            "year-of-sheep-zodiac-chinese",
            "shepherd-king-david-biblical",
            "woolly-economy-medieval-european",
            "scapegoat-paired-yom-kippur-judaic",
            "hajj-sacrifice-islamic",
            "pastoral-foundation-near-eastern",
            "conformity-passive-modern-western",
        ),
    ),
    "horse_galloping": AttributeBundle(
        concept="horse_galloping",
        attributes=(
            "rhythmic", "four-beat", "hooved", "fast", "thunderous",
            "ground-striking", "four-legged", "dust-raising", "equine",
            "percussive", "regular-cadence", "sustained",
            "urgency-evoking", "charge-anticipation", "primal-rush",
        ),
        cultural_attributes=(
            "warrior-charge-medieval-european",
            "four-horsemen-apocalypse-christian",
            "mongol-cavalry-ancestor",
            "plains-buffalo-hunt-indigenous-north-american",
            "cossack-charge-russian",
            "valkyrie-ride-norse",
            "hippodrome-chariot-roman",
            "kentucky-derby-american",
            "ride-of-rohirrim-modern-mythic",
            "cavalry-charge-pan-cultural",
        ),
    ),
    "horse_trotting": AttributeBundle(
        concept="horse_trotting",
        attributes=(
            "rhythmic", "two-beat", "hooved", "moderate-pace", "four-legged",
            "steady", "ground-striking", "equine", "cobblestone-clatter",
            "dust-light", "regular-cadence", "sustained",
            "pastoral-presence", "journey-anticipation", "leisurely-mood",
        ),
        cultural_attributes=(
            "carriage-aristocratic-victorian",
            "dressage-noble-european",
            "steppe-rider-mongol",
            "milk-cart-rural-pan-cultural",
            "horse-and-buggy-pre-industrial-western",
            "paso-fino-iberian-latin-american",
            "year-of-horse-zodiac-chinese",
            "pony-express-american",
            "tinker-traveller-irish-romani",
            "cab-hansom-victorian-english",
        ),
    ),
    # ─── Batch 2: birds ─────────────────────────────────────────────────
    "dove_cooing": AttributeBundle(
        concept="dove_cooing",
        attributes=(
            "small-bird", "gray-or-white", "two-legged", "feathered", "perched",
            "ground-feeding", "plump", "soft-coo", "repetitive-call",
            "low-pitched", "gentle-flight", "pair-bonded",
            "peaceful-presence", "soothing-mood", "melancholic-resonance",
        ),
        cultural_attributes=(
            "peace-symbol-pan-cultural",
            "holy-spirit-christian",
            "noah-flood-dove-biblical",
            "aphrodite-bird-greek",
            "mourning-dove-elegiac-american",
            "ishtar-deity-mesopotamian",
            "soul-of-departed-mediterranean",
            "monogamous-virtue-pan-cultural",
            "pigeon-feast-roman",
            "olive-branch-judaic-flood",
        ),
    ),
    "duck_quacking": AttributeBundle(
        concept="duck_quacking",
        attributes=(
            "webbed-feet", "two-legged", "waterfowl", "feathered", "swimming",
            "flying", "dabbling", "bill-flat", "quack-loud", "gregarious",
            "mottled-plumage", "waddling",
            "comedic-figure", "pastoral-presence", "children-delight",
        ),
        cultural_attributes=(
            "peking-duck-cuisine-chinese",
            "mandarin-duck-marriage-east-asian",
            "donald-duck-cartoon-american",
            "sitting-duck-easy-target-western",
            "ducklings-fairy-tale-european",
            "hunting-quarry-pan-cultural",
            "ugly-duckling-andersen-danish",
            "foie-gras-controversy-french",
            "decoy-craft-folk-american",
            "lame-duck-political-american",
        ),
    ),
    "goose_honking": AttributeBundle(
        concept="goose_honking",
        attributes=(
            "large-waterfowl", "two-legged", "webbed-feet", "long-necked",
            "feathered", "migratory", "V-formation", "loud-honk", "territorial",
            "gregarious", "gray-or-white", "watchful",
            "vigilant-presence", "urgency-evoking", "sentinel-alarm",
        ),
        cultural_attributes=(
            "capitoline-geese-roman",
            "christmas-goose-feast-western",
            "golden-goose-fairy-tale-european",
            "mother-goose-nursery-rhyme-english",
            "silly-goose-foolishness-western",
            "great-cackler-cosmic-egg-egyptian",
            "saraswati-hamsa-mount-hindu",
            "barnacle-goose-lent-loophole-medieval-european",
            "wild-goose-chase-idiom-english",
            "migratory-omen-pan-cultural",
        ),
    ),
    "hen_clucking": AttributeBundle(
        concept="hen_clucking",
        attributes=(
            "domestic-bird", "two-legged", "feathered", "scratching-ground",
            "plump", "egg-laying", "social-flock", "brood-protective",
            "comb-wattle", "ground-feeder", "vocal", "short-flight",
            "maternal-presence", "pastoral-calm", "fussy-aura",
        ),
        cultural_attributes=(
            "easter-egg-symbol-christian",
            "year-of-rooster-shared-zodiac-chinese",
            "mother-hen-protective-idiom-english",
            "chicken-and-egg-paradox-philosophical",
            "hen-pecked-husband-folk-european",
            "sacrificial-bird-various-african",
            "divination-haruspex-roman",
            "domestic-virtue-pastoral",
            "ovum-mundi-creation-egg-myth-pan-cultural",
            "foghorn-leghorn-cartoon-american",
        ),
    ),
    "songbird_singing": AttributeBundle(
        concept="songbird_singing",
        attributes=(
            "small-bird", "two-legged", "feathered", "perched", "melodious",
            "varied-call", "dawn-chorus", "territorial", "courtship-display",
            "light-flight", "varied-plumage", "vocal",
            "joyful-presence", "awakening-mood", "melodic-uplift",
        ),
        cultural_attributes=(
            "muse-inspiration-greek",
            "orpheus-taming-beasts-with-song-greek",
            "nightingale-keats-romantic-english",
            "philomela-myth-greek",
            "blackbird-omen-celtic",
            "lark-ascending-pastoral-english",
            "divination-augury-roman",
            "songbird-as-soul-mesoamerican",
            "canary-coalmine-warning-modern-western",
            "dawn-blessing-pan-cultural",
        ),
    ),
    "turkey_gobbling": AttributeBundle(
        concept="turkey_gobbling",
        attributes=(
            "large-bird", "two-legged", "feathered", "ground-dwelling", "plump",
            "fan-tailed", "wattled", "vocal-gobble", "brown-plumage",
            "scratching-ground", "social-flock", "short-flight",
            "comedic-figure", "pastoral-presence", "abundance-aura",
        ),
        cultural_attributes=(
            "thanksgiving-feast-american",
            "christmas-feast-modern-western",
            "benjamin-franklin-proposal-american",
            "mesoamerican-sacred-bird-aztec",
            "foolish-bird-idiom-english",
            "cold-turkey-withdrawal-idiom-english",
            "harvest-abundance-american",
            "ottoman-name-confusion-historical",
            "feathered-headdress-pueblo-indigenous",
            "thanksgiving-pardon-presidential-american",
        ),
    ),
    # ─── Batch 3: marine + body vocalizations ──────────────────────────
    "dolphin_clicking": AttributeBundle(
        concept="dolphin_clicking",
        attributes=(
            "marine-mammal", "sleek-bodied", "intelligent", "social-pod",
            "echolocating", "blowhole", "dorsal-fin", "gray-or-blue",
            "fast-swimmer", "breaching", "vocal", "click-train",
            "playful-presence", "curious-aura", "intelligent-mystery",
        ),
        cultural_attributes=(
            "apollo-delphi-greek",
            "sailor-rescuer-greek-roman",
            "intelligence-marvel-modern-western",
            "boto-pink-shape-shifter-amazonian",
            "divine-messenger-polynesian",
            "sacred-taniwha-water-guardian-maori",
            "naval-mascot-american",
            "dolphin-therapy-modern-belief",
            "flipper-tv-icon-american",
            "spirit-animal-pan-cultural-modern",
        ),
    ),
    "laughter": AttributeBundle(
        concept="laughter",
        attributes=(
            "human-vocal", "social", "rhythmic", "breathy", "contagious",
            "vocal-cord-vibration", "ascending-falling", "varied-pitch",
            "sudden-burst", "sustained-or-bursty", "multi-syllabic",
            "facial-display",
            "joyful-presence", "amusement-trigger", "social-bonding",
        ),
        cultural_attributes=(
            "best-medicine-idiom-english",
            "laughing-buddha-budai-buddhist",
            "comedy-tragedy-mask-greek-roman",
            "jester-court-medieval-european",
            "shaman-laughter-indigenous-pan-cultural",
            "divine-laughter-creation-myth-pan-cultural",
            "schadenfreude-german",
            "laugh-track-modern-western",
            "warding-laughter-buddhist-japanese",
            "last-laugh-revenge-folkloric",
        ),
    ),
    "crying": AttributeBundle(
        concept="crying",
        attributes=(
            "human-vocal", "tear-shedding", "breath-catching", "throat-tight",
            "social-distress", "varied-pitch", "sustained-or-bursty",
            "facial-distress", "vocal-modulation", "body-shaking",
            "voiced-or-silent", "often-mournful",
            "grief-trigger", "vulnerability-display", "empathy-evoking",
        ),
        cultural_attributes=(
            "mourning-ritual-pan-cultural",
            "weeping-madonna-christian",
            "jeremiah-lamentations-judaic",
            "professional-mourners-mediterranean",
            "dirge-funeral-celtic",
            "crocodile-tears-deception-idiom-english",
            "banshee-wail-celtic",
            "kwan-yin-compassion-tears-buddhist",
            "weeping-philosopher-heraclitus-greek",
            "tears-of-the-sun-incan",
        ),
    ),
    "scream": AttributeBundle(
        concept="scream",
        attributes=(
            "human-vocal", "loud", "high-pitched", "sudden-burst", "piercing",
            "alarm-trigger", "sustained", "vocal-cord-strain", "terror-induced",
            "involuntary", "social-alarm", "facial-distress",
            "terror-trigger", "alarm-signal", "primal-fear-response",
        ),
        cultural_attributes=(
            "harpy-shriek-greek",
            "edvard-munch-painting-modern-european",
            "horror-film-trope-modern-western",
            "fight-or-flight-modern-psychology",
            "valkyrie-cry-norse",
            "la-llorona-folkloric-mexican",
            "war-cry-pan-cultural",
            "kabuki-mie-shout-japanese",
            "primal-scream-therapy-modern-western",
            "psalm-22-eli-eli-biblical-christian",
        ),
    ),
    "sneezing": AttributeBundle(
        concept="sneezing",
        attributes=(
            "human-vocal", "sudden-burst", "nasal-explosive", "involuntary",
            "reflex", "breath-expulsion", "vocal-cord-vibration",
            "irritant-triggered", "social-recognizable", "percussive",
            "momentary", "body-shaking",
            "relief-trigger", "surprise-evoking", "contagion-warning",
        ),
        cultural_attributes=(
            "god-bless-you-superstition-pan-cultural",
            "soul-escape-belief-pan-cultural",
            "plague-omen-medieval-european",
            "gesundheit-blessing-german",
            "divination-bone-sneeze-japanese-folkloric",
            "allergy-marker-modern-medical",
            "hay-fever-modern-medical",
            "tzu-gezunt-blessing-judaic",
            "talked-about-superstition-folkloric",
            "ahchoo-cartoon-onomatopoeia-american",
        ),
    ),
    "coughing": AttributeBundle(
        concept="coughing",
        attributes=(
            "human-vocal", "throat-irritation", "sudden-burst", "breath-expulsion",
            "harsh", "vocal-cord-strain", "involuntary-or-voluntary",
            "illness-marker", "percussive", "dry-or-wet", "contagious",
            "body-shaking",
            "illness-signal", "discomfort-display", "social-discomfort",
        ),
        cultural_attributes=(
            "tuberculosis-romantic-tragedy-victorian-european",
            "cough-medicine-folk-remedy-pan-cultural",
            "attention-getting-cough-idiom-pan-cultural",
            "plague-bringer-medieval-european",
            "smoker-cough-modern-western",
            "whooping-cough-disease-pan-cultural",
            "polite-cough-throat-clearing-modern-western",
            "opium-cough-syrup-victorian-english",
            "throat-clearing-rhetorical-pause-pan-cultural",
            "dramatic-death-cough-trope-victorian-literature",
        ),
    ),
    "yawning": AttributeBundle(
        concept="yawning",
        attributes=(
            "human-vocal", "breath-intake", "jaw-stretching", "sustained", "slow",
            "involuntary", "contagious", "fatigue-marker", "mouth-wide-open",
            "eye-tearing", "body-stretching", "social-spreading",
            "fatigue-display", "contagion-trigger", "drowsy-presence",
        ),
        cultural_attributes=(
            "contagion-empathy-modern-psychology",
            "boredom-display-pan-cultural",
            "dragon-yawning-creation-myth-east-asian",
            "soul-escape-yawn-belief-pan-cultural",
            "hippo-yawn-territorial-naturalist",
            "stretch-and-yawn-yoga-modern-western",
            "devil-mouth-medieval-warning-european",
            "lazy-monday-modern-western",
            "gaping-mouth-fear-warning-pan-cultural",
            "lion-yawn-dominance-savannah-naturalist",
        ),
    ),
    "snoring": AttributeBundle(
        concept="snoring",
        attributes=(
            "human-vocal", "sleep-state", "breath-rhythm", "throat-vibration",
            "low-rumble", "sustained", "involuntary", "varied-pitch",
            "sometimes-loud", "body-relaxed", "nasal-or-oral",
            "partner-disturbing",
            "comedic-figure", "sleep-soundness-marker", "partner-annoyance",
        ),
        cultural_attributes=(
            "rip-van-winkle-american-tale",
            "sleeping-beauty-fairy-tale-european",
            "saw-logs-idiom-english",
            "sleep-apnea-modern-medical",
            "fee-fi-fo-fum-giant-folkloric-english",
            "snoring-village-folklore-pan-cultural",
            "kumbhakarna-epic-sleeper-hindu",
            "narcolepsy-modern-medical",
            "baby-snoring-cute-pan-cultural",
            "comedic-snore-cartoon-pan-cultural",
        ),
    ),
    "hushing": AttributeBundle(
        concept="hushing",
        attributes=(
            "human-vocal", "sibilant", "sustained", "soft", "finger-to-lips",
            "low-volume", "calming-intent", "brief-or-sustained",
            "breath-controlled", "silencing", "peer-directed", "child-directed",
            "calming-presence", "secrecy-aura", "intimate-mood",
        ),
        cultural_attributes=(
            "harpocrates-silence-greek",
            "library-quiet-modern-western",
            "lullaby-soothing-pan-cultural",
            "conspiracy-secret-folkloric",
            "hush-money-bribery-idiom-english",
            "monastic-silence-buddhist-christian",
            "omerta-silence-italian-sicilian",
            "hush-puppy-fried-southern-american",
            "secret-keeper-virtue-pan-cultural",
            "vow-of-silence-monastic-pan-religious",
        ),
    ),
    "heart_beating": AttributeBundle(
        concept="heart_beating",
        attributes=(
            "bodily", "rhythmic", "percussive", "internal", "chest-pulsing",
            "varied-rate", "sustained", "life-marker", "blood-pumping",
            "fight-or-flight-accelerated", "audible-stethoscope", "fast-or-slow",
            "life-presence", "urgency-marker", "vulnerability-evoking",
        ),
        cultural_attributes=(
            "seat-of-soul-pan-cultural",
            "valentine-symbol-modern-western",
            "telltale-heart-poe-american",
            "sacred-heart-jesus-catholic",
            "ka-heart-soul-egyptian",
            "anahata-chakra-hindu",
            "fetal-heartbeat-attachment-modern-psychology",
            "lub-dub-medical-modern-western",
            "heart-on-sleeve-idiom-english",
            "broken-heart-idiom-pan-cultural",
        ),
    ),
    # ─── Batch 4: exclamations + body-vocal residue ────────────────────
    "hmm": AttributeBundle(
        concept="hmm",
        attributes=(
            "human-vocal", "nasal", "sustained", "low-pitched", "breath-controlled",
            "monotone", "contemplative", "mouth-closed", "varied-duration",
            "voiced", "hum-like", "social",
            "contemplative-presence", "hesitation-marker", "deliberation-mood",
        ),
        cultural_attributes=(
            "socratic-aporia-pondering-greek",
            "zen-mu-koan-buddhist",
            "scholar-stroke-beard-pan-cultural",
            "deliberation-pause-modern-western",
            "sufi-zikr-islamic",
            "ahem-throat-clearing-idiom-english",
            "doctor-diagnosis-modern-western",
            "wise-elder-pondering-pan-cultural",
            "mmm-satisfaction-modern-western",
            "umm-filler-word-modern-western",
        ),
    ),
    "ugh": AttributeBundle(
        concept="ugh",
        attributes=(
            "human-vocal", "guttural", "sudden-burst", "brief", "throat-back",
            "voiced", "displeasure-expression", "mouth-pursed", "breath-expulsion",
            "social-recognizable", "monosyllabic", "falling-pitch",
            "disgust-trigger", "weariness-display", "exasperation-signal",
        ),
        cultural_attributes=(
            "monday-morning-modern-western",
            "neanderthal-cliche-modern-cartoon",
            "primal-disgust-darwin-modern-evolutionary",
            "cosmic-weariness-existentialist-modern-french",
            "slavic-disappointment-pan-cultural",
            "charlie-brown-good-grief-peanuts-american",
            "evil-eye-warding-mediterranean",
            "complaint-folk-pan-cultural",
            "miasma-disgust-greek",
            "dirty-job-protest-pan-cultural",
        ),
    ),
    "wow": AttributeBundle(
        concept="wow",
        attributes=(
            "human-vocal", "sustained", "rising-falling", "ascending-pitch",
            "mouth-rounded", "breath-expulsion", "voiced", "sudden-burst",
            "social-recognizable", "monosyllabic", "exclamatory",
            "expressive-faced",
            "amazement-trigger", "awe-evoking", "delight-signal",
        ),
        cultural_attributes=(
            "surfer-slang-american",
            "scooby-doo-cartoon-american",
            "beatnik-jazz-mid-century-american",
            "sublime-aesthetic-burke-romantic-english",
            "owen-wilson-meme-modern-american",
            "sublime-romantic-pan-european",
            "dazzling-revelation-pan-cultural",
            "advertising-pitch-modern-western",
            "wow-signal-seti-modern-scientific",
            "kawaii-delight-japanese",
        ),
    ),
    "stuttering": AttributeBundle(
        concept="stuttering",
        attributes=(
            "human-vocal", "repetition", "blocked-onset", "repeated-syllable",
            "frustration-marker", "involuntary", "sudden-pause",
            "sustained-fragment", "social-recognizable", "varied-duration",
            "voiced", "anxiety-driven",
            "anxiety-display", "vulnerability-evoking", "frustration-signal",
        ),
        cultural_attributes=(
            "kings-speech-george-vi-british-historical",
            "moses-tongue-tied-biblical",
            "demosthenes-pebbles-overcoming-greek",
            "claudius-emperor-historical-roman",
            "taboo-stigma-pan-cultural",
            "anxiety-link-modern-psychology",
            "childhood-developmental-modern-medical",
            "dysfluency-clinical-modern-western",
            "fluency-shame-pan-cultural",
            "porky-pig-cartoon-american",
        ),
    ),
    "kiss": AttributeBundle(
        concept="kiss",
        attributes=(
            "human-vocal", "lips-pursed", "brief", "suction-or-smack",
            "breath-controlled", "intimate", "social-affection",
            "recognizable-onomatopoeia", "plosive", "voiceless-or-voiced",
            "mouth-shape-rounded", "momentary",
            "intimate-presence", "affection-display", "romantic-mood",
        ),
        cultural_attributes=(
            "judas-betrayal-biblical",
            "mistletoe-yule-celtic-norse",
            "hollywood-romance-modern-western",
            "eskimo-nose-rub-inuit",
            "french-romantic-modern-pan-cultural",
            "blow-a-kiss-folk-pan-cultural",
            "sealing-kiss-medieval-european",
            "veneration-kissing-feet-pan-religious",
            "butterfly-kiss-eyelash-folk-pan-cultural",
            "mezuzah-kiss-religious-judaic",
        ),
    ),
    "belching": AttributeBundle(
        concept="belching",
        attributes=(
            "human-vocal", "sudden-burst", "throat-back", "gas-expulsion",
            "deep-pitch", "post-meal", "involuntary-or-voluntary", "brief",
            "social-recognizable", "voiced", "mouth-open", "percussive",
            "comedic-figure", "satisfaction-after-meal", "social-discomfort",
        ),
        cultural_attributes=(
            "compliment-to-chef-folk-mythologized-various",
            "table-manners-violation-western",
            "comedic-trope-modern-western",
            "frat-boy-burp-contest-american",
            "viking-feast-belch-norse-historical",
            "gluttony-vice-christian",
            "satisfaction-marker-pan-cultural",
            "burping-acceptance-pastoral-mongol",
            "gross-out-humor-modern-western",
            "post-meal-alhamdulillah-islamic-some-folk",
        ),
    ),
    "flatulence": AttributeBundle(
        concept="flatulence",
        attributes=(
            "bodily", "gas-expulsion", "sudden-burst", "brief", "posterior",
            "low-pitch", "comedic-trope", "involuntary-or-voluntary", "percussive",
            "social-taboo", "recognizable-onomatopoeia", "momentary",
            "comedic-figure", "social-embarrassment", "taboo-trigger",
        ),
        cultural_attributes=(
            "comedic-staple-pan-cultural",
            "scatological-humor-modern-western",
            "fart-jokes-aristophanes-greek",
            "asher-yatzar-blessing-judaic",
            "devil-stench-medieval-european",
            "pull-my-finger-folk-american",
            "he-gassen-fart-scroll-edo-japanese",
            "fartleberry-medieval-folkloric-english",
            "taboo-public-pan-cultural",
            "gas-relief-modern-medical",
        ),
    ),
    # ─── Batch 5: ingestion ────────────────────────────────────────────
    "biting": AttributeBundle(
        concept="biting",
        attributes=(
            "jaws", "teeth-clamping", "sudden-force", "percussive", "dental",
            "momentary-or-sustained", "mandibular", "tooth-on-flesh",
            "predatory-or-feeding", "oral", "audible-snap", "varied-pressure",
            "aggressive-presence", "hunger-trigger", "primal-feeding-aura",
        ),
        cultural_attributes=(
            "vampire-fangs-gothic-european",
            "snake-bite-medical-emergency-pan-cultural",
            "apple-of-eden-bite-biblical",
            "lyme-disease-tick-bite-modern-medical",
            "ouroboros-snake-biting-tail-egyptian-greek",
            "dog-bite-rabies-folkloric",
            "sharkbite-jaws-modern-american",
            "hickey-love-bite-modern-romantic",
            "wedding-cake-bite-folk-american",
            "predator-prey-pan-cultural",
        ),
    ),
    "brushing_teeth": AttributeBundle(
        concept="brushing_teeth",
        attributes=(
            "bristle-rhythmic", "water-mixed", "foam-producing", "scrubbing",
            "repetitive", "oral-hygiene", "two-handed-or-electric", "sustained",
            "mechanical", "paste-bearing", "morning-or-evening", "intimate-routine",
            "routine-presence", "freshening-mood", "ritual-cleanliness",
        ),
        cultural_attributes=(
            "modern-hygiene-routine-western",
            "miswak-stick-prophet-islamic",
            "ayurvedic-neem-stick-hindu",
            "fluoride-debate-modern-western",
            "dental-school-modern-professional",
            "toothfairy-mythology-western",
            "toothpick-after-meal-pan-cultural",
            "gum-disease-prevention-modern-medical",
            "mouthwash-listerine-modern-american",
            "smile-cosmetic-modern-western",
        ),
    ),
    "drinking": AttributeBundle(
        concept="drinking",
        attributes=(
            "liquid-intake", "mouth-open", "swallowing-action", "throat-action",
            "audible-gulp", "sustained-or-bursty", "lip-pursing", "container-bearing",
            "social-or-solitary", "voiced-or-silent", "smacking-sometimes",
            "satisfaction-marker",
            "thirst-satisfaction-relief", "social-bonding", "refreshment-mood",
        ),
        cultural_attributes=(
            "communion-wine-christian",
            "prohibition-1920s-american",
            "beer-stein-german",
            "sake-ceremony-japanese",
            "kava-ritual-polynesian",
            "tea-ceremony-japanese",
            "mead-feast-norse",
            "alcohol-taboo-islamic",
            "ayahuasca-ceremony-amazonian",
            "libation-pouring-greek-roman",
        ),
    ),
    "eating": AttributeBundle(
        concept="eating",
        attributes=(
            "mouth-chewing", "jaw-rhythmic", "audible-chewing", "swallow-action",
            "mastication", "social-or-solitary", "varied-pace", "social-table",
            "satisfaction-marker", "voiced-or-silent", "food-bearing",
            "oral-multistage",
            "hunger-satisfaction", "social-bonding", "comfort-mood",
        ),
        cultural_attributes=(
            "communion-bread-christian",
            "kosher-laws-judaic",
            "halal-laws-islamic",
            "vegetarian-ahimsa-hindu-buddhist",
            "thanksgiving-feast-american",
            "fine-dining-michelin-modern-western",
            "comfort-food-modern-pan-cultural",
            "sacrament-of-eating-pan-religious",
            "gluttony-vice-christian",
            "ramadan-iftar-islamic",
        ),
    ),
    "swallowing": AttributeBundle(
        concept="swallowing",
        attributes=(
            "throat-contraction", "larynx-rising", "peristaltic", "momentary",
            "audible-gulp", "esophageal", "post-chewing", "oral-to-pharyngeal",
            "voluntary-then-involuntary", "repeatable", "dry-or-wet",
            "voiced-or-silent",
            "tension-relief", "food-passage-marker", "sometimes-difficult-aura",
        ),
        cultural_attributes=(
            "dysphagia-modern-medical",
            "communion-wafer-christian",
            "jonah-whale-swallow-biblical",
            "kronos-eats-children-greek",
            "swallow-pride-idiom-english",
            "swallow-bitter-pill-idiom-english",
            "throat-chakra-vishuddha-hindu",
            "frog-in-throat-idiom-english",
            "choking-hazard-modern-medical",
            "ritual-bitter-herbs-passover-judaic",
        ),
    ),
}
# fmt: on


@dataclass
class ConceptSignature:
    """Cross-linguistic phonological signature aggregated over one concept.

    This is the *fuzzy* version: list of forms keyed by language. Sharpness
    (how much variance across languages) is computable from it but not
    pre-computed here — that's a downstream measurement step.
    """

    concept: str
    n_languages: int
    n_entries: int
    languages: list[str]
    orthographies: list[str]
    ipas: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def signature_for(slug: str, entries: Iterable[AnchorEntry]) -> ConceptSignature:
    rows = [e for e in entries if e.concept == slug]
    langs = sorted({e.language_code for e in rows if e.language_code})
    return ConceptSignature(
        concept=slug,
        n_languages=len(langs),
        n_entries=len(rows),
        languages=langs,
        orthographies=[e.orthography for e in rows if e.orthography],
        ipas=[e.ipa for e in rows if e.ipa],
    )


@dataclass
class AttributeAnchor:
    concept: str
    attribute: str
    cultural: bool
    signature: ConceptSignature
    # Filled in Stage 5: vector position in SAE feature space.
    attribute_embedding: list[float] | None = None
    # Filled in Stage 6: aggregated articulatory-feature vector.
    phonological_features: list[float] | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def build_attribute_anchor_table(
    entries: Iterable[AnchorEntry],
    registry: dict[str, AttributeBundle] = ATTRIBUTE_REGISTRY,
) -> list[AttributeAnchor]:
    entries_list = list(entries)
    out: list[AttributeAnchor] = []
    for bundle in registry.values():
        sig = signature_for(bundle.concept, entries_list)
        for attr in bundle.attributes:
            out.append(
                AttributeAnchor(
                    concept=bundle.concept, attribute=attr, cultural=False, signature=sig
                )
            )
        for attr in bundle.cultural_attributes:
            out.append(
                AttributeAnchor(
                    concept=bundle.concept, attribute=attr, cultural=True, signature=sig
                )
            )
    return out


def write_attribute_anchors(table: Iterable[AttributeAnchor], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in table:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def run_from_jsonl(anchors_jsonl: Path, dest_jsonl: Path) -> Path:
    """Convenience: read anchors-v1.jsonl, write attribute-anchors.jsonl."""
    entries = read_jsonl(anchors_jsonl)
    table = build_attribute_anchor_table(entries)
    write_attribute_anchors(table, dest_jsonl)
    return dest_jsonl
