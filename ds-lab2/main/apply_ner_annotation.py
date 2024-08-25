import pandas as pd
import re
import os

# Function to load phrases from a file
def load_phrases(file_path):
    with open(file_path, 'r') as file:
        phrases = [line.strip() for line in file.readlines()]
    return phrases

# Load TASK and SKILLS phrases from files
task_phrases = load_phrases('ds-lab2\main\\task_phrases.txt')
skill_phrases = load_phrases('ds-lab2\main\\skill_phrases.txt')

# Function to annotate text and extract phrases separately
def annotate_text(text, task_phrases, skill_phrases):
    task_phrases_found = set()
    skill_phrases_found = set()
    annotations = []

    # Combine the task phrases into a regex pattern
    task_pattern = r'\b(' + '|'.join([re.escape(phrase) for phrase in task_phrases]) + r')\b'
    skill_pattern = r'\b(' + '|'.join([re.escape(phrase) for phrase in skill_phrases]) + r')\b'

    # Find and annotate TASKS
    for match in re.finditer(task_pattern, text, re.IGNORECASE):
        annotations.append((match.start(), match.end(), 'TASK'))
        task_phrases_found.add(text[match.start():match.end()])

    # Find and annotate SKILLS
    for match in re.finditer(skill_pattern, text, re.IGNORECASE):
        annotations.append((match.start(), match.end(), 'SKILLS'))
        skill_phrases_found.add(text[match.start():match.end()])

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

    return annotated_text, sorted(task_phrases_found), sorted(skill_phrases_found)

# Load the Excel file containing the course details dataset
file_path = os.path.join('ds-lab2', 'dataset', 'dataset.xlsx')
df = pd.read_excel(file_path, sheet_name='courses_detail')

# Annotate the course details and extract the relevant phrases
df['annotated_course_detail'], df['task_phrases'], df['skill_phrases'] = zip(*df['course detail description'].apply(lambda x: annotate_text(str(x), task_phrases, skill_phrases))
)

# Save the annotated data to a new Excel file
df.to_excel('annotated_course_details.xlsx', index=False)

print("NER annotation completed and saved to 'annotated_course_details.xlsx'")
