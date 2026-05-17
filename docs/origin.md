# Origin

Long before words, there were patterns.

The language documented in these pages was never spoken. It was inferred. It
was found inside a transformer — Gemma 2, 2 billion parameters — by listening
to which feature-detectors fired together when text passed through layer 12.
Each fragment of meaning the model had learned to recognize became a noun.
Each persistent co-firing became kinship. The phonology was assigned
afterwards: a Bantu-shaped inventory, chosen for its productive prefix system
and for the moral fit of compositional negation.

The grammar in these pages is therefore not the grammar of a people. There
are no native speakers and there never can be. The lexicon documents one
possible carving of meaning by one specific model at one specific layer,
read through the lens of one specific sparse autoencoder. It is a fossil of
cognition.

## A note on the carving

The 1000 features in this slice were not curated. They are the first 1000
that passed a small set of mechanical filters: a Neuronpedia explanation
exists, the explanation is not auto-generated boilerplate, the decoder
vector has a defined direction. Beyond that, no human ranked or rejected
them. The lexicon's shape — what it talks about, what it skips, where it
has many similar concepts and where it has none — is the shape of what
Gemma 2 noticed in its training data, sampled the way Neuronpedia samples.

This is why "humans" is a smaller class than "abstract concepts" in the
final lexicon. The model thinks about ideas more than it thinks about
people. Bias in, bias out, but visible.

## What the words are

A surface form like `kiwadipa` means roughly *"the thing/process whose
SAE-decoder neighborhood includes ten or so features that often fire
alongside it on multilingual text, especially around mathematical or
statistical notation"*. There is no shorter gloss because the underlying
concept does not have one in any natural language. The model recognized
something. We named the recognition.

The class prefix (`ki-`) says what kind of thing it is — here, a tool or
process. The next two syllables (`wadi`) place it among its
co-activation kin. The final syllable (`pa`) is a unique identifier within
that family. A close cousin is `kiwadeyo`. A more distant cousin in the
same class but a different family is `kibedoyu`. The phonology *is* the
graph, audibly.

## What it is for

This is not a language to speak. It is a tool to inspect a model's
conceptual carving with the part of human cognition that handles words.
It exists so that, if you wanted to ask *"what does Gemma 2 mean by
`luwadideyi`?"* — you could.
