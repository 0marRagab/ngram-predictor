import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Now your other imports will work
from src.model.ngram_model import NGramModel
class Predictor:
    """
    Responsibility: Accepts a pre-loaded NGramModel and Normalizer, 
    normalizes input text, and returns the top-k predicted next words 
    sorted by probability. Backoff lookup is delegated to NGramModel.lookup().
    """

    def __init__(self, model, normalizer):
        """
        Accept a pre-loaded NGramModel and Normalizer instance.
        """
        self.model = model
        self.normalizer = normalizer

    def normalize(self, text):
        """
        Calls Normalizer.normalize(text) and extracts the last N-1 words.
        """
        clean_text = self.normalizer.normalize(text)
        tokens = clean_text.split()
        # Extract last N-1 words (e.g., last 3 words if N=4)
        return tokens[-(self.model.n - 1):] if tokens else []

    def map_oov(self, context_tokens):
        """
        Replaces words not found in the vocabulary with <UNK>.
        """
        return [word if word in self.model.vocab else "<UNK>" for word in context_tokens]

    def predict_next(self, text, k):
        """
        Orchestrates the prediction pipeline: normalize -> map_oov -> lookup -> rank.
        """
        # 1. Normalize
        context = self.normalize(text)
        
        # 2. Map OOV
        mapped_context = self.map_oov(context)
        
        # 3. Lookup (Delegated to model)
        candidates = self.model.lookup(mapped_context)
        
        if not candidates:
            return []
            
        # 4. Rank by probability (descending) and return top-k keys
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [word for word, prob in sorted_candidates[:k]]

# --- THE TEST CODE STARTS HERE (NO INDENTATION) ---

if __name__ == "__main__":
    # These imports must be inside the if __name__ block to avoid circular imports
    from src.model.ngram_model import NGramModel
    from src.data_prep.normalizer import Normalizer

    # 1. Setup a quick mock model
    test_model = NGramModel(n=4)
    test_model.vocab = {"the", "cat", "sat", "on", "mat", "<UNK>"}
    
    # Fill the model dictionary
    test_model.model["4gram"] = {"the cat sat": {"on": 1.0}}
    test_model.model["1gram"] = {"the": 0.5, "cat": 0.5}

    # 2. Setup instances
    norm = Normalizer()
    pred = Predictor(test_model, norm)

    # 3. Test it!
    print("--- Testing Predictor Standalone ---")
    result = pred.predict_next("The cat sat", k=1)
    print(f"Input: 'The cat sat' -> Result: {result}")