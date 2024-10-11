import stanza
import fastcoref
import re

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

