import torch
import cProfile
import pstats
import io
import time
import yaml
import pickle
from pathlib import Path

from src.model.transformer import Transformer
from src.tokenizer.bpe import BPETokenizer
from src.inference.decoder import inference


def load_model_checkpoint(checkpoint_dir: str) -> str:
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoint_files = list(checkpoint_path.glob("checkpoint_iter_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    checkpoint_files_sorted = sorted(checkpoint_files, key=lambda f: int(f.stem.split('_')[-1]), reverse=True)

    for checkpoint_file in checkpoint_files_sorted:
        try:
            torch.load(str(checkpoint_file), map_location='cpu')
            return str(checkpoint_file)
        except Exception as e:
            print(f"Warning: Skipping corrupted checkpoint {checkpoint_file.name}: {e}\n")
            continue
    raise FileNotFoundError(f"No valid checkpoint files found in {checkpoint_dir}")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    params = {}
    
    if 'parameters' in config:
        for key, value in config['parameters'].items():
            if isinstance(value, dict) and 'value' in value:
                params[key] = value['value']
            else:
                params[key] = value
    else:
        params = config

    return params


def load_model(checkpoint_path: str, config: dict, device: str = "cuda") -> Transformer:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"]

    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    d_model = config['d_model']
    num_heads = config['num_heads']
    d_ff = config['d_ff']
    vocab_size = config['vocab_size']
    context_length = config['context_length']
    num_layers = config['num_layers']
    theta = config['theta']


    model = Transformer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        theta=theta
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def load_tokenizer(tokenizer_cache_path: str) -> BPETokenizer:
    with open(tokenizer_cache_path, "rb") as f:
        data = pickle.load(f)

    tokenizer = BPETokenizer(special_tokens=data.get("special_tokens", []))
    tokenizer.vocabulary = data["vocabulary"]
    tokenizer.reverseVocab = data["reverseVocab"]
    tokenizer.merges = data["merges"]
    tokenizer.sorted_merges = {
        (a, b): i
        for i, (a, b) in enumerate(tokenizer.merges)
    }
    return tokenizer


def run(model, tokenizer, device, prompt, max_tokens=100, temperature=1.0, top_p=None, show_profiler=None):
    profiler = cProfile.Profile()
    start_time = time.perf_counter()

    profiler.enable()
    print(f"Prompt: {repr(prompt)}")
    print("Output: ", end='', flush=True)

    output = None
    prev_output = prompt
    for decoded_text in inference(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device
    ):
        output = decoded_text
        new_text = output[len(prev_output):]
        print(new_text, end='', flush=True)
        prev_output = output

    print()
    profiler.disable()

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    output_tokens = tokenizer.encode(output)
    prompt_tokens = tokenizer.encode(prompt)
    generated_tokens = len(output_tokens) - len(prompt_tokens)
    tokens_per_second = generated_tokens / elapsed if elapsed > 0 else 0

    print()
    print("Stats:")
    print(f"  Prompt tokens: {len(prompt_tokens)}")
    print(f"  Generated tokens: {generated_tokens}")
    print(f"  Latency: {tokens_per_second:.2f} tokens/s")
    print(f"  Time: {elapsed:.3f}s")

    result = {
        "prompt": prompt,
        "output": output,
        "prompt_tokens": len(prompt_tokens),
        "generated_tokens": generated_tokens,
        "time": elapsed,
        "tokens_per_second": tokens_per_second
    }

    s = io.StringIO()
    if show_profiler:
        stats = pstats.Stats(profiler, stream=s)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats(20)
        print(s.getvalue())


        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.strip_dirs()
        stats.sort_stats('tottime')
        stats.print_stats(20)
        print(s.getvalue())
    
    profile_output = "src/observability/benchmarks/inference_profile.stats"
    profiler.dump_stats(profile_output)
    print(f"\nDetailed profiling stats saved to: {profile_output}")
    return result


def main():

    CONFIG_PATH = "src/configs/pretrain.yaml"
    CHECKPOINT_DIR = "src/checkpoints/model/tiny-model"
    TOKENIZER_PATH = "src/tokenizer/cache/owt_train_v50257_st0_cache.pkl"
    DEVICE = "cuda" 

    PROMPT = "What is the meaning of life?"

    MAX_TOKENS = 200
    TEMPERATURE = 0.8
    TOP_P = 0.9

    try:
        config = load_config(CONFIG_PATH)
    except FileNotFoundError as e:
        print(f"Error loading config: {e}")
        return

    try:
        checkpoint_path = load_model_checkpoint(CHECKPOINT_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    try:
        tokenizer = load_tokenizer(TOKENIZER_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    try:
        model = load_model(checkpoint_path, config=config, device=DEVICE)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    run(
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        prompt=PROMPT,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P
    )


if __name__ == "__main__":
    main()
