# text_preprocessing.py

import spacy
import re

# Load spaCy's small English model
nlp = spacy.load('en_core_web_sm')

def lemmatize_text(text):
    """
    Lemmatize the input text to reduce words to their base forms.
    
    :param text: The text to be lemmatized
    :return: Lemmatized text
    """
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def preprocess_text(text):
    """
    Perform all text preprocessing steps on the input text.
    
    :param text: The text to be preprocessed
    :return: Preprocessed text
    """
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove newline characters and extra spaces
    text = lemmatize_text(text)  # Lemmatize the text
    
    # Add any additional preprocessing steps here if needed
    return text
