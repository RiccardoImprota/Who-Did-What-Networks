from .svo_extraction import *
import spacy
import stanza
import warnings

_nlp_spacy = None
_nlp_stanza = None

def get_spacy_nlp():
    global _nlp_spacy
    if _nlp_spacy is None:
        # Suppress specific spaCy warnings during load if needed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _nlp_spacy = spacy.load('en_core_web_trf')
    return _nlp_spacy

def get_stanza_nlp():
    global _nlp_stanza
    if _nlp_stanza is None:
        # Suppress specific stanza warnings during load if needed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,coref', verbose=False)
    return _nlp_stanza


__version__ = "0.1"
