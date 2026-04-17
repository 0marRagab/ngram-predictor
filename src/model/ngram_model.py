import json
import os
from collections import Counter, defaultdict

class NGramModel:
    """Core N-Gram logic for building vocabulary and calculating probabilities."""

    def __init__(self, n=4, unk_threshold=2):
        """Initializes model order and unknown word frequency threshold."""
        self.n = n
        self.unk_threshold = unk_threshold
        self.vocab = set()
        self.model = {f"{i}gram": {} for i in range(1, n + 1)}

    def build_vocab(self, sentences):
        """Creates a set of words that appear at least unk_threshold times."""
        word_counts = Counter()
        for sentence in sentences:
            word_counts.update(sentence.split())
        self.vocab = {word for word, count in word_counts.items() if count >= self.unk_threshold}
        self.vocab.add("<UNK>")

    def _replace_unk(self, tokens):
        """Replaces tokens not in vocabulary with the <UNK> tag."""
        return [word if word in self.vocab else "<UNK>" for word in tokens]

    def build_counts_and_probs(self, sentences):
        """Calculates Maximum Likelihood Estimation probabilities for all N-gram levels."""
        counts = {i: defaultdict(Counter) for i in range(1, self.n + 1)}
        total_words = 0

        for sentence in sentences:
            tokens = self._replace_unk(sentence.split())
            total_words += len(tokens)
            for window_size in range(1, self.n + 1):
                for i in range(len(tokens) - window_size + 1):
                    window = tokens[i : i + window_size]
                    if window_size == 1:
                        counts[1][""][window[0]] += 1
                    else:
                        context = " ".join(window[:-1])
                        target = window[-1]
                        counts[window_size][context][target] += 1

        for word, count in counts[1][""].items():
            self.model["1gram"][word] = count / total_words

        for order in range(2, self.n + 1):
            key = f"{order}gram"
            for context, targets in counts[order].items():
                context_total = sum(targets.values())
                self.model[key][context] = {w: c / context_total for w, c in targets.items()}

    def lookup(self, context_tokens):
        """Implements backoff logic: searches from highest order down to unigram."""
        for length in range(self.n - 1, -1, -1):
            key = f"{length + 1}gram"
            if length == 0:
                return self.model.get(key, {})
            context_str = " ".join(context_tokens[-length:])
            if context_str in self.model.get(key, {}):
                return self.model[key][context_str]
        return {}

    def save_model(self, model_path, vocab_path):
        """Serializes the model and vocabulary to JSON files."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(self.model, f, indent=4)
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(list(self.vocab), f, indent=4)