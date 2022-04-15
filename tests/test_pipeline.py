from spacy import Language

from spacy_partial_tagger.pipeline import PartialEntityRecognizer


def test_partial_ner_pipeline_initializable(nlp: Language) -> None:
    pipe = nlp.create_pipe("partial_ner")

    assert isinstance(pipe, PartialEntityRecognizer)
