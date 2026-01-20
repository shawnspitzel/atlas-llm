# Overview
AtlasLLM is a from-scratch implementation of a Transformer-based large language model. This includes from-scratch implementations of the AdamOptimizer, Multi-Head Attention, FlashAttention, RoPE Embeddings, and so on. The goal with this project is to create a production-grade language model with all necessary systems optimizations.

To keep up to date with progress, check TODO.md.

# Usage
## Data Preprocessing / Tokenization
In order to tokenize data, first store your training and validation data splits in 
````src/data/raw/train```` and ````src/data/raw/val```` respectively. From here, run:

````uv run python -m src.training.preprocess_config --config src/configs/preprocess.yaml````

This will tokenize your training and validation datasets, and store them within ````src/data/tokenized/````.