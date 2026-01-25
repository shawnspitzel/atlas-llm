# Atlas LLM

AtlasLLM is a from-scratch implementation of a Transformer-based large language model. Below is a list containing some of the following features I've implemented from scratch:

## Features
- Multi-head attention with RoPE embeddings
- SwiGLU feed-forward networks
- RMSNorm normalization
- Custom AdamW optimizer
- Distributed training (Multi-GPU)
- Optimizer state sharding (ZeRO-1)
- BPE tokenizer
- Top-p sampling
- Temperature-controlled generation
- Weights & Biases observability

## Training

### Data Preprocessing / Tokenization
In order to tokenize data, first store your training and validation data splits in
`src/data/raw/train` and `src/data/raw/val` respectively. From here, run:

```bash
uv run python -m src.tokenizer.preprocess_config --config src/configs/preprocess.yaml
```

This will tokenize your training and validation datasets, and store them within `src/data/tokenized/`.

### Configurations
This project makes heavy use of configurations for training, of which some sample pretrain configurations can be found in ```src/configs```. A configuration is necessary to pretrain. If you'd like to use W&B to monitor progress, simply run:
```bash
wandb sweep src/configs/<your_config>.yaml
wandb agent <your_agent_id>
```
This will handle pretraining for you, defaulting to single GPU pretraining. If you'd like to manually pretrain, consider the options below.

## Manual Training
### Single GPU
```bash
uv run python -m src.training.pretrain --config src/configs/<your_config>.yaml
```

### Distributed Training (Multi-GPU)

```bash
torchrun --nproc_per_node=2 -m src.training.distributed_pretrain --config src/configs/<your_config>.yaml
```

## Inference
If you'd like to test how a trained model performs, navigate to ```src/inference/inference_script.py``` and hook up the latest model checkpoint into it's load path. From there, run:
```bash
uv run python src/inference/inference_script.py
```

## Project Structure

```
src/
├── model/          # Transformer architecture
├── training/       # Training scripts
├── inference/      # Generation and decoding
├── tokenizer/      # BPE tokenizer
├── systems/        # Systems optimizations
├── observability/  # Profiling & benchmarking
└── configs/        # Training configs
```

See [TODO.md](TODO.md) for current progress and planned work.
