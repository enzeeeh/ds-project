# rule_based_ner.py

import spacy
from spacy.matcher import PhraseMatcher

def load_phrases(file_path):
    with open(file_path, 'r') as file:
        phrases = file.read().splitlines()
    return phrases

def create_phrase_matcher(nlp, task_phrases, skill_phrases):
    matcher = PhraseMatcher(nlp.vocab)
    
    task_patterns = [nlp.make_doc(text) for text in task_phrases]
    skill_patterns = [nlp.make_doc(text) for text in skill_phrases]
    
    matcher.add("TASK", task_patterns)
    matcher.add("SKILL", skill_patterns)
    
    return matcher

def apply_matcher(nlp, matcher, text):
    doc = nlp(text)
    matches = matcher(doc)
    
    entities = []
    for match_id, start, end in matches:
        span = doc[start:end]
        entities.append((span.text, nlp.vocab.strings[match_id]))
    return entities

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    
    task_phrases = load_phrases("ds-lab2\main\\task_phrases.txt")
    skill_phrases = load_phrases("ds-lab2\main\\skill_phrases.txt")
    
    matcher = create_phrase_matcher(nlp, task_phrases, skill_phrases)
    
    # Example text
    text = "This course covers cybersecurity and teaches students how to implement firewalls."
    entities = apply_matcher(nlp, matcher, text)
    
    print("Entities found:", entities)
