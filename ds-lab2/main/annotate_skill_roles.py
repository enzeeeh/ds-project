import pandas as pd
import re

# Function to load phrases from a file
def load_phrases(file_path):
    with open(file_path, 'r') as file:
        phrases = [line.strip() for line in file.readlines()]
    return phrases

# Load TASK and SKILLS phrases from files
task_phrases = load_phrases('task_phrases.txt')
skill_phrases = load_phrases('skill_phrases.txt')

# Function to annotate text
def annotate_text(text, task_phrases, skill_phrases):
    annotations = []
    
    # Combine the task phrases into a regex pattern
    task_pattern = r'\b(' + '|'.join([re.escape(phrase) for phrase in task_phrases]) + r')\b'
    skill_pattern = r'\b(' + '|'.join([re.escape(phrase) for phrase in skill_phrases]) + r')\b'
    
    # Find and annotate TASKS
    for match in re.finditer(task_pattern, text, re.IGNORECASE):
        annotations.append((match.start(), match.end(), 'TASK'))

    # Find and annotate SKILLS
    for match in re.finditer(skill_pattern, text, re.IGNORECASE):
        annotations.append((match.start(), match.end(), 'SKILLS'))

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

    return annotated_text

# Example: Annotate text from a single column
def annotate_column(df, column_name, task_phrases, skill_phrases):
    return df[column_name].apply(lambda x: annotate_text(str(x), task_phrases, skill_phrases))

# Load the Excel file
df = pd.read_excel('skill_roles.xlsx')

# Annotate the mission, main tasks, and key skills columns
df['Annotated Mission'] = annotate_column(df, 'mission', task_phrases, skill_phrases)
df['Annotated Main Tasks'] = annotate_column(df, 'main tasks', task_phrases, skill_phrases)
df['Annotated Key Skills'] = annotate_column(df, 'key skills', task_phrases, skill_phrases)

# Save the annotated data back to Excel
df.to_excel('annotated_skill_roles.xlsx', index=False)

print("Annotation completed and saved to 'annotated_skill_roles.xlsx'")
