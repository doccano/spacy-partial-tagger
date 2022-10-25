# spacy-partial-tagger

This is a library to build a CRF tagger for a partially annotated dataset in spaCy. You can build your own NER tagger only from dictionary. The algorithm of this tagger is based on Effland and Collins. (2021).

## Overview

![The overview of spacy-partial-tagger](https://raw.githubusercontent.com/doccano/spacy-partial-tagger/main/images/overview.png)

## Dataset Preparation

Prepare spaCy binary format file. This library expects tokenization is character-based.
For more detail about spaCy binary format, see [this page](https://spacy.io/api/data-formats#training).

You can prepare your own dataset with [spaCy's entity ruler](https://spacy.io/usage/rule-based-matching#entityruler) as follows:

```py
import spacy
from spacy.tokens import DocBin
from spacy_partial_tagger.tokenizer import CharacterTokenizer


nlp = spacy.blank("en")
nlp.tokenizer = CharacterTokenizer(nlp.vocab)  # Use a character-based tokenizer

patterns = [{"label": "LOC", "pattern": "Tokyo"}, {"label": "LOC", "pattern": "Japan"}]
ruler = nlp.add_pipe("entity_ruler")
ruler.add_patterns(patterns)

doc = nlp("Tokyo is the capital of Japan.")

doc_bin = DocBin()
doc_bin.add(doc)

# Replace /path/to/data.spacy with your own path
doc_bin.to_disk("/path/to/data.spacy")
```

In the example above, entities are extracted by character-based matching. However, in some cases, character-based matching may not be suitable (e.g., the element symbol `na` for sodium matches name). In such cases, token-based matching can be used as follows:

```py
import spacy
from spacy.tokens import DocBin
from spacy_partial_tagger.tokenizer import CharacterTokenizer

text = "Selegiline - induced postural hypotension in Parkinson's disease: a longitudinal study on the effects of drug withdrawal."
patterns = [
    {"label": "Chemical", "pattern": [{"LOWER": "selegiline"}]},
    {"label": "Disease", "pattern": [{"LOWER": "hypotension"}]},
    {
        "label": "Disease",
        "pattern": [{"LOWER": "parkinson"}, {"LOWER": "'s"}, {"LOWER": "disease"}],
    },
]

# Add an entity ruler to the pipeline.
nlp = spacy.blank("en")
ruler = nlp.add_pipe("entity_ruler")
ruler.add_patterns(patterns)

# Extract entities from the text.
doc = nlp(text)
entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

# Create a DocBin object.
nlp = spacy.blank("en")
nlp.tokenizer = CharacterTokenizer(nlp.vocab)
doc_bin = DocBin()
doc = nlp.make_doc(text)
doc.ents = [
    doc.char_span(start, end, label=label) for start, end, label in entities
]
doc_bin.add(doc)
doc_bin.to_disk("/path/to/data.spacy")
```

## Training

Train your model as follows:

```sh
python -m spacy train config.cfg --output outputs --paths.train /path/to/train.spacy --paths.dev /path/to/dev.spacy --gpu-id 0
```

You could download `config.cfg` [here](https://github.com/tech-sketch/spacy-partial-tagger/blob/main/config.cfg).
Or you could setup your own. This library would train models through spaCy. If you are not familiar with spaCy's config file format, please check the [documentation](https://spacy.io/usage/training#config).

Don't forget to replace `/path/to/train.spacy` and `/path/to/dev.spacy` with your own.

## Evaluation

Evaluate your model as follows:

```sh
python -m spacy evaluate outputs/model-best /path/to/test.spacy --gpu-id 0
```

Don't forget to replace `/path/to/test.spacy` with your own.

## Installation

```sh
pip install spacy-partial-tagger
```

If you use M1 Mac, you might have problems installing `fugashi`. In that case, please try `brew install mecab` before the installation.

## References

- Thomas Effland and Michael Collins. 2021. [Partially Supervised Named Entity Recognition via the Expected Entity Ratio Loss](https://aclanthology.org/2021.tacl-1.78/). _Transactions of the Association for Computational Linguistics_, 9:1320–1335.
