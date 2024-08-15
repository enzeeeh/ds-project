import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
import en_core_web_sm
nlp = en_core_web_sm.load()

nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self, study_program, skillset):
        self.study_program = study_program
        self.skillset = skillset
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = nlp

    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def clean_text(self, text):
        """Remove HTML tags, special characters, numbers, and multiple spaces, then lowercase"""
        text = re.sub(r'<[^<>]*>', '', text)
        text = re.sub(r'[^\w\s]|[\d]', '', text)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    def preprocess(self, df, columns):
        """Apply cleaning and tokenizing"""
        for col in columns:
            df[f'preprocessing_{col}'] = df[col].apply(self.clean_text)
            df[f'preprocessing_{col}'] = df[f'preprocessing_{col}'].apply(lambda x: self.nlp(x))
            df[f'clean_tokens_{col}'] = df[f'preprocessing_{col}'].apply(
                lambda x: [self.lemmatizer.lemmatize(token.text, self.get_wordnet_pos(token.text)) for token in x if token.text not in STOP_WORDS])
        return df

    def combine_skills(self):
        """Combine skill, mission, tasks, knowledge into one list of tokens"""
        combined_columns = ['clean_tokens_mission', 'clean_tokens_main_tasks', 'clean_tokens_key_skills', 'clean_tokens_key_knowledge']
        self.skillset['clean_combined_token_skill'] = self.skillset[combined_columns].apply(lambda row: sum(row, []), axis=1)

    def execute(self):
        study_columns = ['study_program_name', 'description']
        skill_columns = ['profile_title', 'mission', 'main_tasks', 'key_skills', 'key_knowledge']

        self.study_program = self.preprocess(self.study_program, study_columns)
        self.skillset = self.preprocess(self.skillset, skill_columns)

        for col in study_columns:
            self.study_program[f'clean_document_{col}'] = self.study_program[f'clean_tokens_{col}'].apply(lambda x: " ".join(x))

        self.skillset['clean_document_profile_title'] = self.skillset['clean_tokens_profile_title'].apply(lambda x: " ".join(x))

        self.combine_skills()
        self.skillset['token_count_combined_skill'] = self.skillset['clean_combined_token_skill'].apply(len)
        self.skillset['clean_document_combined_skill'] = self.skillset['clean_combined_token_skill'].apply(lambda tokens: " ".join(tokens))

        return self.study_program, self.skillset