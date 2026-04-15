import json
import os
from collections import Counter, defaultdict

class NGramModel:
    def __init__(self, n=4, unk_threshold=2):
        self.n = n
        self.unk_threshold = unk_threshold
        self.vocab = set()
        # Initialize keys: "1gram" through "Ngram"
        self.model = {f"{i}gram": {} for i in range(1, n + 1)}

    def build_vocab(self, sentences):
        word_counts = Counter()
        for sentence in sentences:
            word_counts.update(sentence.split())
        
        self.vocab = {word for word, count in word_counts.items() if count >= self.unk_threshold}
        self.vocab.add("<UNK>")
        print(f"Vocab built. Size: {len(self.vocab)}")

    def _replace_unk(self, tokens):
        return [word if word in self.vocab else "<UNK>" for word in tokens]

    def build_counts_and_probs(self, sentences):
        # 1. Initialize temporary counts dictionary
        # counts[n] stores { context: { target: frequency } }
        counts = {i: defaultdict(Counter) for i in range(1, self.n + 1)}
        total_words = 0

        # 2. Count Occurrences
        for sentence in sentences:
            tokens = self._replace_unk(sentence.split())
            total_words += len(tokens)
            
            for window_size in range(1, self.n + 1):
                for i in range(len(tokens) - window_size + 1):
                    window = tokens[i : i + window_size]
                    
                    if window_size == 1:
                        # Unigram case: empty context
                        counts[1][""][window[0]] += 1
                    else:
                        # context is everything but the last word
                        context = " ".join(window[:-1])
                        target = window[-1]
                        counts[window_size][context][target] += 1

        # 3. Calculate Probabilities and store in self.model
        # 1gram: 0-word context
        for word, count in counts[1][""].items():
            self.model["1gram"][word] = count / total_words

        # 2gram to Ngram: Mapping window_size directly to key name
        # A 4gram uses a window of 4 words (3 context + 1 target)
        for order in range(2, self.n + 1):
            key = f"{order}gram"
            for context, targets in counts[order].items():
                context_total = sum(targets.values())
                self.model[key][context] = {
                    word: c / context_total for word, c in targets.items()
                }
        
        print("Probabilities calculated for all N-gram levels.")

    def lookup(self, context_tokens):
        """Performs backoff lookup using the last N-1 tokens."""
        # Start from highest order (n-1 tokens for n-gram) down to 0
        for length in range(self.n - 1, -1, -1):
            key = f"{length + 1}gram"
            
            if length == 0:
                return self.model.get(key, {})
            
            context_str = " ".join(context_tokens[-length:])
            
            if context_str in self.model.get(key, {}):
                return self.model[key][context_str]
        
        return {}

    def save_model(self, model_path, vocab_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(self.model, f, indent=4)
        
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(list(self.vocab), f, indent=4)
        print(f"Model saved to {model_path}")