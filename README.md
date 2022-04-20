# spacy-partial-tagger


## Installation

```
poetry install
```


## Data preparation

Download pre-processed files:

```sh
curl -LO "https://raw.githubusercontent.com/teffland/ner-expected-entity-ratio/main/data/conll2003/eng/entity.train_r0.5_p0.9.jsonl"
curl -LO "https://raw.githubusercontent.com/teffland/ner-expected-entity-ratio/main/data/conll2003/eng/entity.dev.jsonl"
curl -LO "https://raw.githubusercontent.com/teffland/ner-expected-entity-ratio/main/data/conll2003/eng/entity.test.jsonl"
```

Convert data format:

```sh
python -m spacy_partial_tagger.convert --input entity.train_r0.5_p0.9.jsonl --output train.spacy
python -m spacy_partial_tagger.convert --input entity.dev.jsonl --output dev.spacy
python -m spacy_partial_tagger.convert --input entity.test.jsonl --output test.spacy
```

## Training

```sh
python -m spacy train config.cfg --output outputs
```

## Evaluation

```sh
python -m spacy evaluate outputs/model-best test.spacy
```
