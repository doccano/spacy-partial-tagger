# spacy-partial-tagger

This is a library to build a CRF tagger for a partially annotated dataset in spaCy. You can build your own NER tagger only from dictionary. The algorithm of this tagger is based on Effland and Collins. (2021).

## Overview

![The overview of spacy-partial-tagger](https://raw.githubusercontent.com/doccano/spacy-partial-tagger/main/images/overview.png)

## Dataset Preparation

Prepare spaCy binary format file to train your tagger.
If you are not familiar with spaCy binary format, see [this page](https://spacy.io/api/data-formats#training).

You can prepare your own dataset with [spaCy's entity ruler](https://spacy.io/usage/rule-based-matching#entityruler) as follows:

```py
import spacy
from spacy.tokens import DocBin


nlp = spacy.blank("en")

patterns = [{"label": "LOC", "pattern": "Tokyo"}, {"label": "LOC", "pattern": "Japan"}]
ruler = nlp.add_pipe("entity_ruler")
ruler.add_patterns(patterns)

doc = nlp("Tokyo is the capital of Japan.")

doc_bin = DocBin()
doc_bin.add(doc)

# Replace /path/to/data.spacy with your own path
doc_bin.to_disk("/path/to/data.spacy")
```

## Training

Train your tagger as follows:

```sh
python -m spacy train config.cfg --output outputs --paths.train /path/to/train.spacy --paths.dev /path/to/dev.spacy --gpu-id 0
```

This library is implemented as [a trainable component](https://spacy.io/usage/layers-architectures#components) in spaCy,
so you could control the training setting via spaCy's configuration system.
We provide you the default configuration file [here](https://github.com/tech-sketch/spacy-partial-tagger/blob/main/config.cfg).
Or you could setup your own. If you are not familiar with spaCy's config file format, please check the [documentation](https://spacy.io/usage/training#config).

Don't forget to replace `/path/to/train.spacy` and `/path/to/dev.spacy` with your own.

## Evaluation

Evaluate your tagger as follows:

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
