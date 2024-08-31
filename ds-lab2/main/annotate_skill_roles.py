import pandas as pd
import spacy
import re
import os
import json

# Load spaCy's small English model
nlp = spacy.load('en_core_web_sm')

# Function to load phrases from a file
def load_phrases(file_path):
    with open(file_path, 'r') as file:
        phrases = [line.strip() for line in file.readlines()]
    return phrases

# Load TASK and SKILLS phrases from files
task_phrases = load_phrases('ds-lab2\main\\task_phrases.txt')
skill_phrases = load_phrases('ds-lab2\main\\skill_phrases.txt')

# Function to preprocess text (removes new lines and any extra spaces)
def preprocess_text(text):
    text = text.replace('\n', ' ').replace('\r', ' ')  # Remove new lines and carriage returns
    text = re.sub(r',', '', text)  # Remove commas
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space
    return text

# Function to annotate text and extract phrases separately
def annotate_text(text, task_phrases, skill_phrases):
    task_phrases_found = []
    skill_phrases_found = []
    annotations = []

    # Combine the task phrases into a regex pattern
    task_pattern = r'\b(' + '|'.join([re.escape(phrase) for phrase in task_phrases]) + r')\b'
    skill_pattern = r'\b(' + '|'.join([re.escape(phrase) for phrase in skill_phrases]) + r')\b'

    # Find and annotate TASKS
    for match in re.finditer(task_pattern, text, re.IGNORECASE):
        annotations.append((match.start(), match.end(), 'TASK'))
        task_phrases_found.append(text[match.start():match.end()])

    # Find and annotate SKILLS
    for match in re.finditer(skill_pattern, text, re.IGNORECASE):
        annotations.append((match.start(), match.end(), 'SKILLS'))
        skill_phrases_found.append(text[match.start():match.end()])

    # Sort annotations by start index
    annotations = sorted(annotations, key=lambda x: x[0])

    # Annotate the text with labels
    annotated_text = text
    offset = 0
    for start, end, label in annotations:
        annotated_text = (annotated_text[:start + offset] +
                          f"[{annotated_text[start + offset:end + offset]}]({label})" +
                          annotated_text[end + offset:])
        offset += len(f"[({label})]")  # Adjust offset due to added characters

    return annotated_text, task_phrases_found, skill_phrases_found

# Load the Excel file containing the skill roles dataset
file_path2 = os.path.join('ds-lab2', 'dataset', 'enisa_skill_set.xlsx')
df = pd.read_excel(file_path2, sheet_name='skill_set')

# Initialize sets to collect unique task and skill phrases across all columns
all_task_phrases = set()
all_skill_phrases = set()

# Apply the preprocessing to remove new lines when loading the text columns
df['mission'] = df['mission'].apply(preprocess_text)
df['main_tasks'] = df['main_tasks'].apply(preprocess_text)
df['key_skills'] = df['key_skills'].apply(preprocess_text)
df['key_knowledge'] = df['key_knowledge'].apply(preprocess_text)

# Annotate and collect task and skill phrases separately for each profile
df['annotated_mission'], mission_task_phrases, mission_skill_phrases = zip(*df['mission'].apply(lambda x: annotate_text(str(x), task_phrases, skill_phrases)))
df['annotated_main_tasks'], main_tasks_task_phrases, main_tasks_skill_phrases = zip(*df['main_tasks'].apply(lambda x: annotate_text(str(x), task_phrases, skill_phrases)))
df['annotated_key_skills'], key_skills_task_phrases, key_skills_skill_phrases = zip(*df['key_skills'].apply(lambda x: annotate_text(str(x), task_phrases, skill_phrases)))
df['annotated_key_knowledge'], key_knowledge_task_phrases, key_knowledge_skill_phrases = zip(*df['key_knowledge'].apply(lambda x: annotate_text(str(x), task_phrases, skill_phrases)))

# Combine the task and skill phrases for each skill profile (per row)
df['unique_task_phrases'] = [
    sorted(set(mission) | set(main) | set(key_skill) | set(key_knowledge))
    for mission, main, key_skill , key_knowledge in zip(mission_task_phrases, main_tasks_task_phrases, key_skills_task_phrases, key_knowledge_task_phrases)
]

df['unique_skill_phrases'] = [
    sorted(set(mission) | set(main) | set(key_skill) | set(key_knowledge))
    for mission, main, key_skill , key_knowledge in zip(mission_skill_phrases, main_tasks_skill_phrases, key_skills_skill_phrases, key_knowledge_skill_phrases)
]

# Convert the DataFrame to a dictionary
data_dict = df.to_dict(orient='records')

# Save the annotated data to a new Excel file
df.to_excel('annotated_skill_roles.xlsx', index=False)

# Save the annotated data to a JSON file
output_file = 'annotated_skill_roles.json'
with open(output_file, 'w') as json_file:
    json.dump(data_dict, json_file, indent=4)

print(f"NER annotation completed and saved to '{output_file}'")