from .svo_extraction import *
import spacy
import stanza
import warnings
from .nlp_utils import get_spacy_nlp, get_stanza_nlp

_nlp_spacy = get_spacy_nlp()
_nlp_stanza = get_stanza_nlp()

__version__ = "0.1"
