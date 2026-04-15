import argparse
import os
import json
from dotenv import load_dotenv

# Import your modules
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor

def main():
    # 0. Setup Configuration
    load_dotenv("config/.env")
    
    parser = argparse.ArgumentParser(description="NGram Text Predictor CLI")
    parser.add_argument("--step", choices=["dataprep", "model", "inference", "all"], required=True)
    args = parser.parse_args()

    # Match your .env variable names exactly
    raw_dir = os.getenv("TRAIN_RAW_DIR")
    model_path = os.getenv("MODEL")
    vocab_path = os.getenv("VOCAB")
    train_tokens_path = os.getenv("TRAIN_TOKENS") # Matches your .env
    
    # Cast numerical values
    n_order = int(os.getenv("NGRAM_ORDER", 4))
    top_k = int(os.getenv("TOP_K", 3))
    # Note: Use UNK_THRESHOLD if you want to pass it to NGramModel(unk_threshold=...)
    unk_limit = int(os.getenv("UNK_THRESHOLD", 3))

    # 1. Instantiate Core Objects
    norm = Normalizer()
    model = NGramModel(n=n_order, unk_threshold=unk_limit)

    # --- PIPELINE LOGIC ---

    # STEP: Data Prep
    if args.step in ["dataprep", "all"]:
        print(f"Loading files from {raw_dir}...")
        raw_text = norm.load(raw_dir) 
        
        print("Processing and tokenizing sentences...")
        sentences = norm.process_document(raw_text)
        
        # Save to the path defined in TRAIN_TOKENS
        norm.save(sentences, train_tokens_path)
        print(f"Saved {len(sentences)} sentences to {train_tokens_path}")

    # STEP: Model Training
    if args.step in ["model", "all"]:
        if not os.path.exists(train_tokens_path):
            print(f"Error: {train_tokens_path} not found. Run 'dataprep' first.")
            return

        with open(train_tokens_path, "r", encoding="utf-8") as f:
            sentences = f.read().splitlines()
            
        print(f"Building {n_order}-gram model...")
        model.build_vocab(sentences)
        model.build_counts_and_probs(sentences)
        model.save_model(model_path, vocab_path)
        print("Model training complete.")

    # STEP: Inference
    if args.step in ["inference", "all"]:
        if not model.model:
            with open(model_path, 'r') as f: model.model = json.load(f)
            with open(vocab_path, 'r') as f: model.vocab = set(json.load(f))
        
        predictor = Predictor(model, norm)
        
        print("\n" + "="*30)
        print("SHERLOCK HOLMES PREDICTOR READY")
        print("Type 'quit' to exit.")
        print("="*30)

        while True:
            user_input = input("\n> ").strip()
            if user_input.lower() == 'quit':
                break
            if not user_input:
                continue

            predictions = predictor.predict_next(user_input, k=top_k)
            print(f"Predictions: {predictions}")

if __name__ == "__main__":
    main()