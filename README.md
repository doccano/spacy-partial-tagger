# spacy-partial-tagger

This is a CRF tagger for partially annotated dataset in spaCy. You can build your 
own NER tagger only from dictionary. The implementation of this tagger is based on Effland and Collins. (2021).

## Dataset

Prepare spaCy binary format file. This library expects tokenization is character-based.
For more detail about spaCy binary format, see [this page](https://spacy.io/api/data-formats#training).

```py
import spacy
from spacy.tokens import DocBin
from spacy_partial_tagger.tokenizer import CharacterTokenizer


nlp = spacy.blank("en")
nlp.tokenizer = CharacterTokenizer(nlp.vocab)

patterns = [{"label": "LOC", "pattern": "Tokyo"}, {"label": "LOC", "pattern": "Japan"}]
ruler = nlp.add_pipe("entity_ruler")
ruler.add_patterns(patterns)

doc = nlp("Tokyo is the capital of Japan.")


doc_bin = DocBin()
doc_bin.add(doc)

doc_bin.to_disk("/path/to/dataset")
```

## Training

```sh
python -m spacy train config.cfg --output outputs --paths.train train.spacy --paths.dev dev.spacy 
```

## Evaluation

```sh
python -m spacy evaluate outputs/model-best test.spacy
```

## Installation

```
pip install spacy-partial-tagger
```

## References

- Thomas Effland and Michael Collins. 2021. [Partially Supervised Named Entity Recognition via the Expected Entity Ratio Loss](https://aclanthology.org/2021.tacl-1.78/). _Transactions of the Association for Computational Linguistics_, 9:1320–1335.
