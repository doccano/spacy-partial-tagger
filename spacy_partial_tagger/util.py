import catalogue
from spacy.util import registry

registry.label_indexers = catalogue.create(  # type:ignore
    "spacy", "label_indexers", entry_points=True
)
