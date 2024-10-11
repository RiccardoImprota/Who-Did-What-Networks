
def _valences(language="english"):

    if language == "english":
        from valence.english import _positive, _negative, _ambivalent


    elif language == "italian":
        from valence.italian import _positive, _negative, _ambivalent

    return _positive, _negative, _ambivalent