N-Gram Text Predictor: Sherlock Holmes Edition
A modular Python-based Natural Language Processing (NLP) pipeline that processes text from Project Gutenberg, builds a statistical N-Gram language model with Stupid Backoff, and provides an interactive CLI for next-word prediction. This project demonstrates core NLP concepts including text normalization, sentence tokenization, and Out-Of-Vocabulary (OOV) handling using <UNK> tokens.

🛠️ Requirements
Python Version: 3.9+

Environment Manager: Anaconda / Miniconda (recommended)

Dependencies: Managed via requirements.txt (includes NLTK and python-dotenv)

⚙️ Setup & Installation
Clone the Repository:
git clone <your-repository-url>
cd ngram-predictor

Create and Activate Anaconda Environment:
conda create -n ngram_env python=3.9
conda activate ngram_env

Install Dependencies:
pip install -r requirements.txt

Populate Configuration:
Create a config/.env file based on the project requirements. Ensure it contains the following keys:

TRAIN_RAW_DIR: Path to the folder containing raw .txt files.

TRAIN_TOKENS: Path to save processed sentences.

MODEL: Path for the output .json model.

VOCAB: Path for the output .json vocabulary.

NGRAM_ORDER: The value of N (e.g., 4).

Prepare Data:
Download your corpus (e.g., Sherlock Holmes stories from Project Gutenberg) as .txt files and place them into the folder specified in your TRAIN_RAW_DIR variable.

🚀 Usage
The project is controlled via main.py using the --step argument to trigger different parts of the pipeline.

Full Pipeline (Recommended): Run everything from raw text to prediction mode.
python main.py --step all

Data Preparation: Clean, normalize, and tokenize the raw text files.
python main.py --step dataprep

Model Training: Build the N-Gram counts and calculate probabilities.
python main.py --step model

Inference: Launch the interactive CLI to get next-word predictions.
python main.py --step inference

📂 Project Structure
.
├── config/
│   └── .env                # Environment variables and paths
├── data/                   # Generated artifacts (model.json, vocab.json)
├── src/
│   ├── data_prep/
│   │   └── normalizer.py   # Text cleaning and tokenization logic
│   ├── model/
│   │   └── ngram_model.py  # N-Gram counting and backoff logic
│   ├── inference/
│   │   └── predictor.py    # Prediction orchestration
│   ├── evaluation/         # Model performance metrics
│   └── ui/                 # UI components
├── tests/                  # Unit tests for core logic
├── main.py                 # Main entry point for the CLI
├── requirements.txt        # List of project dependencies
└── README.md               # Project documentation

Final Pro-Tip:
Once you paste this into your README.md, make sure to:

Save the file.

Commit and push: ```bash
git add README.md
git commit -m "docs: finalize README with setup, usage, and structure"
git push origin main


