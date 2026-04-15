import os
import re
import nltk

class Normalizer:
    """
    This class handles the preparation of the text corpus. 
    """

    def __init__(self):
        """Initializes the Normalizer and ensures NLTK data is available."""
        nltk.download('punkt')

    def load(self, folder_path):
        """
        Loads, strips, and combines all .txt files.
        """
        combined_text = ""
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                # Using 'with' is good practice for file handling
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                    # Strip headers immediately for each individual file
                    clean_content = self.strip_gutenberg(raw_content)
                    combined_text += clean_content + " "
        return combined_text
    
    def strip_gutenberg(self, text):
        """Removes the Project Gutenberg header and footer."""
        start_marker = "START OF THE PROJECT GUTENBERG"
        end_marker = "END OF THE PROJECT GUTENBERG"
        
        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            # Jump to the next line after the marker to avoid leftover 'EBOOK TEST'
            actual_start = text.find("\n", start_idx)
            return text[actual_start:end_idx].strip()
            
        return text

    def lowercase(self, text):
        return text.lower()

    def remove_punctuation(self, text):
        return re.sub(r'[^\w\s]', '', text)

    def remove_numbers(self, text):
        return re.sub(r'\d+', '', text)

    def remove_whitespace(self, text):
        return " ".join(text.split())

    def normalize(self, text):
        """Cleans a single string (sentence)."""
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        return text

    # --- NEW MASTER METHOD ---
    def process_document(self, raw_text):
        """
        This is the core logic:
        1. Tokenize into sentences while periods/punctuation still exist.
        2. Normalize each sentence individually.
        """
        # A. Find sentences using the periods
        sentences = self.sentence_tokenize(raw_text)
        
        # B. Clean each sentence separately
        cleaned_list = []
        for s in sentences:
            clean_s = self.normalize(s)
            if clean_s: # Only add if it's not empty
                cleaned_list.append(clean_s)
                
        return cleaned_list

    def sentence_tokenize(self, text):
        return nltk.sent_tokenize(text)

    def word_tokenize(self, sentence):
        return nltk.word_tokenize(sentence)

    def save(self, sentences, filepath):
        """Writes processed sentences to a file, one per line."""
        target_dir = os.path.dirname(filepath)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
            
        with open(filepath, 'w', encoding='utf-8') as f:
            for s in sentences:
                f.write(s + "\n")

# --- IMPROVED TEST BLOCK ---
if __name__ == "__main__":
    norm = Normalizer()
    input_folder = "data_test"
    output_file = "output/processed_test.txt"
    
    print("Step 1: Loading and Stripping individual files...")
    raw_text = norm.load(input_folder)
    
    print("Step 2: Processing Document (Tokenizing then Normalizing)...")
    # This replaces the old Step 3 and 4 to fix the one-line bug
    final_sentences = norm.process_document(raw_text)
    
    print(f"Step 3: Saving {len(final_sentences)} sentences to {output_file}...")
    norm.save(final_sentences, output_file)
    
    print("\n--- DONE! ---")