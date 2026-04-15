import os
import re
import nltk

class Normalizer:
    """Handles the cleaning, stripping, and tokenization of the text corpus."""

    def __init__(self):
        """Initializes the Normalizer and downloads necessary NLTK datasets."""
        nltk.download('punkt')

    def load(self, folder_path):
        """Loads and combines all .txt files from a directory into one string."""
        combined_text = ""
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                    clean_content = self.strip_gutenberg(raw_content)
                    combined_text += clean_content + " "
        return combined_text
    
    def strip_gutenberg(self, text):
        """Removes Project Gutenberg headers/footers based on standard markers."""
        start_marker = "START OF THE PROJECT GUTENBERG"
        end_marker = "END OF THE PROJECT GUTENBERG"
        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker)
        if start_idx != -1 and end_idx != -1:
            actual_start = text.find("\n", start_idx)
            return text[actual_start:end_idx].strip()
        return text

    def lowercase(self, text):
        """Converts text to lowercase."""
        return text.lower()

    def remove_punctuation(self, text):
        """Removes all punctuation marks from the string."""
        return re.sub(r'[^\w\s]', '', text)

    def remove_numbers(self, text):
        """Removes all digits from the text."""
        return re.sub(r'\d+', '', text)

    def remove_whitespace(self, text):
        """Collapses multiple whitespaces into a single space."""
        return " ".join(text.split())

    def normalize(self, text):
        """Executes the full cleaning pipeline on a single string."""
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        return text

    def process_document(self, raw_text):
        """Tokenizes raw text into sentences, then normalizes each sentence."""
        sentences = self.sentence_tokenize(raw_text)
        cleaned_list = [self.normalize(s) for s in sentences if self.normalize(s)]
        return cleaned_list

    def sentence_tokenize(self, text):
        """Splits raw text into a list of sentences using NLTK."""
        return nltk.sent_tokenize(text)

    def word_tokenize(self, sentence):
        """Splits a sentence into a list of individual word tokens."""
        return nltk.word_tokenize(sentence)

    def save(self, sentences, filepath):
        """Saves a list of strings to a file, one per line."""
        target_dir = os.path.dirname(filepath)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            for s in sentences:
                f.write(s + "\n")