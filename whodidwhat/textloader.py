import stanza
import re


def text_preparation(text,clean=True):
    """
    Prepares the text for further processing by cleaning it and resolving coreferences.
    
    Parameters:
    text (str): The input text to prepare.

    Returns:
    str: The prepared text with coreferences resolved.
    """
    
    # Clean the text
    cleaned_text = clean_text(text)

    # Solve coreferences
    resolved_text = solve_coreferences(cleaned_text, coref_solver='stanza')

    return resolved_text

def clean_text(text):
    """
    Cleans a given text by removing double spaces and words inside parentheses.
    
    Parameters:
    text (str): The input string to clean.
    
    Returns:
    str: The cleaned text without double spaces and words inside parentheses.
    """
    
    # Step 1: Remove anything inside parentheses including the parentheses
    # The regex '\(.*?\)' matches any content inside parentheses and the parentheses themselves.
    cleaned_text = re.sub(r'\(.*?\)', '', text)
    
    # Step 2: Replace double spaces with single spaces
    # This step is repeated until all double spaces are reduced to single spaces
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
    
    # Step 3: Strip leading and trailing spaces (if any)
    cleaned_text = cleaned_text.strip()

    return cleaned_text

def solve_coreferences(text, coref_solver='stanza'):
    """
    Resolves coreferences in a given text using either the Stanza library or fastcoref.

    Args:
        text (str): The input text to resolve coreferences in.
    Return:
        str: The text with coreferences resolved.
    
    CURRENTLY SUPPORTS STANZA ONLY
    """

    if coref_solver!='stanza':
        raise ValueError("Only Stanza coreference solver is supported at the moment.")
    
    if coref_solver=='stanza':
        # Load the Stanza pipeline
        stanzanlp = stanza.Pipeline(lang='en', processors='tokenize,coref')
        # Process the text
        doc = stanzanlp(text)
        output_text = stanza_solve_coreferences(doc)
    
    return output_text


def stanza_solve_coreferences(doc):
    """
    Replaces coreferent mentions with their representative texts in the text reconstructed from the doc.

    Args:
        doc (stanza.Document): The processed document with coreference information.

    Returns:
        str: The text with coreferences resolved.
    """
    # Use doc.text to get the original text
    original_text = doc.text

    # List to hold the replacements (start_char, end_char, representative_text)
    replacements = []

    # Dictionary to keep track of active mentions per coreference chain
    active_mentions = {}

    # Iterate over sentences and words to build mentions
    for sentence in doc.sentences:
        for word in sentence.words:
            word_start = word.start_char
            word_end = word.end_char

            if word.coref_chains:
                # Get the coref_attachment with the lowest chain_idx
                min_coref_attachment = min(word.coref_chains, key=lambda x: x.chain.index)
                chain_idx = min_coref_attachment.chain.index
                rep_text = min_coref_attachment.chain.representative_text

                if word.text in rep_text:
                  continue

                # Start of a mention
                if min_coref_attachment.is_start:
                    active_mentions[chain_idx] = {
                        'start_char': word_start,
                        'end_char': word_end,
                        'rep_text': rep_text
                    }
                # Continuation of a mention
                elif chain_idx in active_mentions:
                    active_mentions[chain_idx]['end_char'] = word_end

                # End of a mention
                if min_coref_attachment.is_end:
                    if chain_idx in active_mentions:
                        mention = active_mentions[chain_idx]
                        # Record the mention span and its representative text
                        replacements.append((
                            mention['start_char'],
                            mention['end_char'],
                            mention['rep_text']
                        ))
                        # Remove the mention from active mentions
                        del active_mentions[chain_idx]
            else:
                # Word is not part of any coreference chain; nothing to do
                pass

    # Sort replacements in reverse order to prevent index shifting
    replacements.sort(key=lambda x: x[0], reverse=True)

    # Apply replacements to the original text
    resolved_text = original_text
    for start_char, end_char, rep_text in replacements:
        resolved_text = resolved_text[:start_char] + rep_text + resolved_text[end_char:]

    return resolved_text

