# WB_Knowledge_Base
This repository contains an approach to creating a knowledge base information retrieval system.

## Project Overview

The system provides two levels of retrieval:
- **Baseline Model**: Uses TF-IDF and cosine similarity.
- **Optimized Model**: Uses FastText embeddings, a Sentence Transformer model fine-tuned on domain-specific terminology, and an optional re-ranking with a cross-encoder.

The repository enables users to evaluate the optimized model’s retrieval performance by using fine-tuned parameters without requiring full re-training or inference setup.

## Repository Structure

```
knowledge-based-retrieval/
├── data/
│   ├── train_data.csv            # Training data with questions and relevant chunks
│   ├── chunks.csv                # Knowledge base chunks for retrieval
├── models/
│   ├── baseline_model.py         # TF-IDF baseline code
│   ├── optimized_model.py        # FastText and Sentence Transformer code
│   └── fine_tuned_params.pth     # Fine-tuned Sentence Transformer model parameters
├── requirements.txt              # Dependencies for the project
├── README.md                     # Project overview and instructions
└── scripts/
    ├── inference.py              # Script to test the optimized model
    └── evaluate.py               # Script for calculating recall scores
```

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Required Models**:
   - Spacy’s Russian language model:
     ```bash
     python -m spacy download ru_core_news_sm
     ```
   - FastText Vectors (run the following in the terminal to download and extract):
     ```bash
     wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz
     gunzip -k cc.ru.300.bin.gz
     ```
   - Store `cc.ru.300.bin` in the `data/` folder.

3. **Configure Weights & Biases (wandb)**:
   - This project uses [Weights & Biases](https://wandb.ai/) for tracking during model fine-tuning.
   - Set up your Weights & Biases API key:
     ```bash
     wandb login
     ```
   - If you don’t have an API key, create an account on [wandb.ai](https://wandb.ai/) to obtain one.
   - For the purposes of chescking my work, you can use my API key: 25296f77e8a2b6ceaa2806bb4ab94f8090a159c1

4. **Run the Baseline Model**:
   ```bash
   python models/baseline_model.py
   ```

5. **Evaluate the Optimized Model**:
   - Run the evaluation script to calculate Recall@k metrics:
     ```bash
     python scripts/evaluate.py
     ```

6. **Test Retrieval with Optimized Model**:
   - Run the `inference.py` script to input queries and retrieve relevant knowledge base chunks:
     ```bash
     python scripts/inference.py
     ```

## Models Used

- **TF-IDF + Cosine Similarity**: Baseline model for basic retrieval.
- **FastText Vectors**: For word embeddings based on the pre-trained Russian FastText model.
- **Sentence Transformers**: Fine-tuned on domain-specific terminology to capture semantic relevance. Fine-tuning is tracked via wandb.
- **Cross-Encoder (Optional)**: Re-ranks candidate results for improved retrieval accuracy.

## Requirements

All dependencies are listed in `requirements.txt`. Run the following command to install:
```bash
pip install -r requirements.txt
```

## Evaluation

- **Recall@k Calculation**: The `evaluate.py` script provides Recall@1, Recall@3, and Recall@5 scores based on the sampled test set. Works with the optimal model only. For the baseline model, the metrics are calculated directly in its file.
- **Inference Example**: You can run sample queries to see top-k retrieved chunks using the optimized model.
