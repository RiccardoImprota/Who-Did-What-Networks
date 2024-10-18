import spacy
import stanza
import warnings
from whodidwhat.resources import _valences


_nlp_spacy = None
_nlp_stanza = None



def spacynlp(text):
    """
    Process the text using the spaCy NLP pipeline.
    """
    nlp = get_spacy_nlp()

    return nlp(text)


def get_spacy_nlp():
    global _nlp_spacy
    if _nlp_spacy is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _nlp_spacy = spacy.load('en_core_web_trf')
    return _nlp_spacy

def get_stanza_nlp():
    global _nlp_stanza
    if _nlp_stanza is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,coref', verbose=False)
    return _nlp_stanza



def compute_valence(text):
    """
    Compute the valence of a given text based on positive and negative words.

    Args:
        text (str): The text to analyze.

    Returns:
        str: 'positive', 'negative', 'contrasting', or 'neutral'
    """
    positive, negative, ambivalent = _valences('english')
    doc = spacynlp(text.lower())
    pos_count = 0
    neg_count = 0

    for token in doc:
        if token.text in ambivalent:
            continue  # Ignore ambivalent words

        # Check if the token is in positive or negative sets
        if token.text in positive or token.text in negative:

            negated = False
            # Check ancestors for negation

            # Check immediate ancestors for negation
            if token.head.dep_ == 'neg':
                negated = True

            for headchild in token.head.children:
                if headchild.dep_ == 'neg':
                    negated = True

            # Check if the token itself is negated
            if any(child.dep_ == 'neg' for child in token.children):
                negated = True


            ## Check if the token itself has a negation dependency
            #for ancestor in token.ancestors:
            #    if any(child.dep_ == 'neg' for child in ancestor.children):
            #        negated = True
            #        break

            if token.text in positive:
                if negated:
                    neg_count += 1  # Invert positive to negative
                else:
                    pos_count += 1
            elif token.text in negative:
                if negated:
                    pos_count += 1  # Invert negative to positive
                else:
                    neg_count += 1

    # Determine overall valence
    if pos_count > 0 and neg_count > 0:
        return 'contrasting'
    elif pos_count > 0:
        return 'positive'
    elif neg_count > 0:
        return 'negative'
    else:
        return 'neutral'