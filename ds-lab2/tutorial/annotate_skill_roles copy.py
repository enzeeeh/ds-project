import pandas as pd
import spacy
import re
import os
import json
from text_preprocessing import preprocess_text 

# Load spaCy's small English model
nlp = spacy.load('en_core_web_sm')

# Function to load phrases from a file
def load_phrases(file_path):
    with open(file_path, 'r') as file:
        phrases = [line.strip() for line in file.readlines()]
    return phrases

# Load phrases from files
methodology_phrases = load_phrases('ds-lab2\\methodology_phrases.txt')
tasks_phrases = load_phrases('ds-lab2\\tasks_phrases.txt')
tools_phrases = load_phrases('ds-lab2\\tools_phrases.txt')
actions_phrases = load_phrases('ds-lab2\\actions_phrases.txt')

# Function to annotate text and extract phrases separately
def annotate_text(original_text, processed_text, methodology_phrases, tasks_phrases, tools_phrases, actions_phrases):
    annotations = []
    found_positions = set()  # To track positions that have already been annotated
    
    # Combine the phrases into regex patterns
    methodology_pattern = r'\b(' + '|'.join([re.escape(phrase) for phrase in methodology_phrases]) + r')\b'
    tasks_pattern = r'\b(' + '|'.join([re.escape(phrase) for phrase in tasks_phrases]) + r')\b'
    tools_pattern = r'\b(' + '|'.join([re.escape(phrase) for phrase in tools_phrases]) + r')\b'
    actions_pattern = r'\b(' + '|'.join([re.escape(phrase) for phrase in actions_phrases]) + r')\b'

    # Function to map processed matches back to original text positions
    def map_to_original(processed_text, original_text, match):
        match_text = match.group()
        # Find the position of the match in the original text (case-insensitive)
        search = re.search(re.escape(match_text), original_text, re.IGNORECASE)
        if search:
            start, end = search.start(), search.end()
            return start, end
        return None, None
    
    # Annotate METHODOLOGY
    for match in re.finditer(methodology_pattern, processed_text, re.IGNORECASE):
        start, end = map_to_original(processed_text, original_text, match)
        if start is not None and end is not None and start not in found_positions:
            annotations.append((start, end, 'METHODOLOGY'))
            found_positions.update(range(start, end))  # Mark this position as annotated

    # Annotate TASKS
    for match in re.finditer(tasks_pattern, processed_text, re.IGNORECASE):
        start, end = map_to_original(processed_text, original_text, match)
        if start is not None and end is not None and start not in found_positions:
            annotations.append((start, end, 'TASK'))
            found_positions.update(range(start, end))  # Mark this position as annotated

    # Annotate TOOLS
    for match in re.finditer(tools_pattern, processed_text, re.IGNORECASE):
        start, end = map_to_original(processed_text, original_text, match)
        if start is not None and end is not None and start not in found_positions:
            annotations.append((start, end, 'TOOLS'))
            found_positions.update(range(start, end))  # Mark this position as annotated

    # Annotate ACTIONS
    for match in re.finditer(actions_pattern, processed_text, re.IGNORECASE):
        start, end = map_to_original(processed_text, original_text, match)
        if start is not None and end is not None and start not in found_positions:
            annotations.append((start, end, 'ACTIONS'))
            found_positions.update(range(start, end))  # Mark this position as annotated

    # Sort annotations by start index and remove overlaps
    annotations = sorted(annotations, key=lambda x: x[0])

    # Annotate the text with labels in the original text
    annotated_text = original_text
    offset = 0
    for start, end, label in annotations:
        annotated_text = (annotated_text[:start + offset] +
                          f"[{annotated_text[start + offset:end + offset]}]({label})" +
                          annotated_text[end + offset:])
        offset += len(f"[({label})]")  # Adjust offset due to added characters

    # Extract and combine unique phrases found
    methodology_phrases_found = [original_text[start:end] for start, end, label in annotations if label == 'METHODOLOGY']
    tasks_phrases_found = [original_text[start:end] for start, end, label in annotations if label == 'TASK']
    tools_phrases_found = [original_text[start:end] for start, end, label in annotations if label == 'TOOLS']
    actions_phrases_found = [original_text[start:end] for start, end, label in annotations if label == 'ACTIONS']

    unique_phrases = {
        "methodology": sorted(set(methodology_phrases_found)),
        "tasks": sorted(set(tasks_phrases_found)),
        "tools": sorted(set(tools_phrases_found)),
        "actions": sorted(set(actions_phrases_found))
    }

    return annotated_text, unique_phrases

# Load the Excel file containing the skill roles dataset
file_path2 = os.path.join('ds-lab2', 'dataset', 'enisa_skill_set.xlsx')
df = pd.read_excel(file_path2, sheet_name='skill_set')

# Preprocess the text in the dataset
df['processed_mission'] = df['mission'].apply(lambda x: preprocess_text(str(x)))
df['processed_main_tasks'] = df['main_tasks'].apply(lambda x: preprocess_text(str(x)))
df['processed_key_skills'] = df['key_skills'].apply(lambda x: preprocess_text(str(x)))

# Annotate and collect methodology, tasks, tools, and actions phrases separately for each profile
df['annotated_mission'], mission_phrases = zip(*df.apply(lambda row: annotate_text(row['mission'], row['processed_mission'], methodology_phrases, tasks_phrases, tools_phrases, actions_phrases), axis=1))
df['annotated_main_tasks'], main_tasks_phrases = zip(*df.apply(lambda row: annotate_text(row['main_tasks'], row['processed_main_tasks'], methodology_phrases, tasks_phrases, tools_phrases, actions_phrases), axis=1))
df['annotated_key_skills'], key_skills_phrases = zip(*df.apply(lambda row: annotate_text(row['key_skills'], row['processed_key_skills'], methodology_phrases, tasks_phrases, tools_phrases, actions_phrases), axis=1)
)

# Extract and combine the unique phrases for each skill profile
df['unique_methodology_phrases'] = [
    sorted(set(mission['methodology'] + main['methodology'] + key['methodology']))
    for mission, main, key in zip(mission_phrases, main_tasks_phrases, key_skills_phrases)
]

df['unique_task_phrases'] = [
    sorted(set(mission['tasks'] + main['tasks'] + key['tasks']))
    for mission, main, key in zip(mission_phrases, main_tasks_phrases, key_skills_phrases)
]

df['unique_tools_phrases'] = [
    sorted(set(mission['tools'] + main['tools'] + key['tools']))
    for mission, main, key in zip(mission_phrases, main_tasks_phrases, key_skills_phrases)
]

df['unique_actions_phrases'] = [
    sorted(set(mission['actions'] + main['actions'] + key['actions']))
    for mission, main, key in zip(mission_phrases, main_tasks_phrases, key_skills_phrases)
]

# Convert the DataFrame to a dictionary
data_dict = df.to_dict(orient='records')

# Save the annotated data to a new Excel file
df.to_excel('annotated_skill_roles_new.xlsx', index=False)

# Save the annotated data to a JSON file
output_file = 'annotated_skill_roles_new.json'
with open(output_file, 'w') as json_file:
    json.dump(data_dict, json_file, indent=4)

print(f"NER annotation completed and saved to '{output_file}'")
