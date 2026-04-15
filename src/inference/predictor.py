class Predictor:
    """Orchestrates next-word prediction using a model and a normalizer."""

    def __init__(self, model, normalizer):
        """Accepts pre-instantiated model and normalizer (Dependency Injection)."""
        self.model = model
        self.normalizer = normalizer

    def normalize(self, text):
        """Cleans input text and returns the last N-1 context tokens."""
        clean_text = self.normalizer.normalize(text)
        tokens = clean_text.split()
        return tokens[-(self.model.n - 1):] if tokens else []

    def map_oov(self, context_tokens):
        """Converts tokens in the context to <UNK> if they are outside the vocabulary."""
        return [word if word in self.model.vocab else "<UNK>" for word in context_tokens]

    def predict_next(self, text, k):
        """Predicts top-k next words by running the normalization and lookup pipeline."""
        context = self.normalize(text)
        mapped_context = self.map_oov(context)
        candidates = self.model.lookup(mapped_context)
        
        if not candidates:
            return []
            
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [word for word, prob in sorted_candidates[:k]]