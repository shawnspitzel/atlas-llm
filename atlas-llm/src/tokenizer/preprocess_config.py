import argparse
import yaml
from pathlib import Path
from src.tokenizer.preprocess_data import preprocess_data


def run_from_config(config_path: str):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    train_file = config.get('train_file')
    if not train_file:
        raise ValueError("train_file is required in config")

    val_file = config.get('val_file', None)
    vocab_size = config.get('vocab_size', 50257)
    output_dir = config.get('output_dir', None)
    num_workers = config.get('num_workers', 4)

    print(f"Loading configuration from: {config_path}")
    print(f"Training file: {train_file}")
    if val_file:
        print(f"Validation file: {val_file}")
    print(f"Vocab size: {vocab_size}")
    print(f"Workers: {num_workers}")
    print()

    preprocess_data(
        train_file=train_file,
        val_file=val_file,
        vocab_size=vocab_size,
        output_dir=output_dir,
        num_workers=num_workers
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Usage: uv run src.training.preprocess_config --config src/configs/preprocess.yaml"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )

    args = parser.parse_args()
    run_from_config(args.config)
