
import pandas as pd
import re
import json

def convert_to_spacy_format(text, annotated_text):
    entities = []
    pattern = r'\[(.*?)\]\((.*?)\)'
    
    found_phrases = set()
    
    # print(f"Original Text: {text}")
    # print(f"Annotated Text: {annotated_text}")
    
    for match in re.finditer(pattern, annotated_text):
        phrase = match.group(1)
        label = match.group(2).upper()
        
        start = text.find(phrase)
        end = start + len(phrase)
        
        # print(f"Found Phrase: {phrase}, Label: {label}, Start: {start}, End: {end}")
        
        if start != -1 and (start, end, label) not in found_phrases:
            entities.append((start, end, label))
            found_phrases.add((start, end, label))
    
    # print(f"Entities: {entities}")
    
    return text, {"entities": entities}

# Load and process the annotated profile roles dataset
file_path_roles = 'ds-lab2\\annotated_skill_roles.xlsx'
df_annotated_roles = pd.read_excel(file_path_roles)

train_data_roles = []
columns_to_process_roles = ['mission', 'main_tasks', 'key_skills']
annotated_columns_roles = ['annotated_mission', 'annotated_main_tasks', 'annotated_key_skills']

for _, row in df_annotated_roles.iterrows():
    for text_col, annotated_col in zip(columns_to_process_roles, annotated_columns_roles):
        text = row[text_col]
        annotated_text = row[annotated_col]
        if pd.notna(text) and pd.notna(annotated_text):
            train_data_roles.append(convert_to_spacy_format(text, annotated_text))

# Save the profile roles data to a JSON file
output_file_path_roles = 'annotated_skill_roles_spacy.json'
with open(output_file_path_roles, 'w') as f:
    json.dump(train_data_roles, f, indent=4)

# Load and process the annotated course descriptions dataset
file_path_courses = 'ds-lab2\\annotated_course_details.xlsx'
df_annotated_courses = pd.read_excel(file_path_courses)

train_data_courses = []

for _, row in df_annotated_courses.iterrows():
    text = row['course detail description']
    annotated_text = row['annotated_course_detail']
    if pd.notna(text) and pd.notna(annotated_text):
        train_data_courses.append(convert_to_spacy_format(text, annotated_text))

# Save the course descriptions data to a JSON file
output_file_path_courses = 'annotated_course_details_spacy.json'
with open(output_file_path_courses, 'w') as f:
    json.dump(train_data_courses, f, indent=4)

print(f"Data has been saved to {output_file_path_roles} and {output_file_path_courses}")
