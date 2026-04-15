N-Gram Text Predictor: Sherlock Holmes Edition
A modular Python-based Natural Language Processing (NLP) pipeline that processes text from Project Gutenberg, builds a statistical N-Gram language model with Stupid Backoff, and provides an interactive CLI for next-word prediction.

🚀 Features
Modular Pipeline: Separate steps for data preparation, model training, and inference.

Stupid Backoff Algorithm: Automatically falls back from 4-grams down to unigrams to ensure prediction coverage.

OOV Handling: Maps rare words to an <UNK> token based on a configurable frequency threshold.

Zero Hardcoding: All file paths and hyperparameters are managed via environment variables.

📂 Project Structure
Plaintext
.
├── config/
│   └── .env                # Configuration variables (Paths, N-order, etc.)
├── data/                   # Git-ignored directory for datasets and models
├── src/
│   ├── data_prep/
│   │   └── normalizer.py   # Text cleaning and sentence tokenization
│   ├── model/
│   │   └── ngram_model.py  # N-Gram counting and backoff logic
│   ├── inference/
│   │   └── predictor.py    # Prediction orchestration and OOV mapping
├── main.py                 # Main CLI entry point
├── requirements.txt        # Pinned project dependencies
└── README.md               # Project documentation
🛠️ Setup & Installation
Create Anaconda Environment:

Bash
conda create -n ngram_env python=3.9
conda activate ngram_env
Install Dependencies:

Bash
pip install -r requirements.txt
Configure Environment Variables:
Ensure your config/.env file is populated. Key variables include:

TRAIN_RAW_DIR: Path to raw text files.

TRAIN_TOKENS: Path to save processed sentences.

MODEL: Path to save the .json model file.

📖 Usage
Run the project using the main.py entry point with the --step flag.

Run the Full Pipeline (End-to-End)
Bash
python main.py --step all
Individual Steps
Data Preparation: python main.py --step dataprep

Model Training: python main.py --step model

Inference (Interactive): python main.py --step inference

🧪 Technical Implementation
Normalization: We ensure nltk.sent_tokenize is called before punctuation removal to preserve sentence boundaries.

Backoff Logic: The NGramModel.lookup() method handles the recursive search from N-1 tokens down to 0, serving as the single source of truth for backoff logic.

Dependency Injection: The Predictor class is instantiated in main.py by passing in existing instances of Normalizer and NGramModel.