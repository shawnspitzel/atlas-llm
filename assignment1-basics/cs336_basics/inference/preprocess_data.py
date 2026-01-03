import argparse
import numpy as np
import os
from pathlib import Path
from multiprocessing import Pool
import cProfile
import pstats
import gc
import pickle
import tempfile
from tqdm import tqdm
from cs336_basics.tokenizer.bpe import BPETokenizer
from cs336_basics.inference.pretokenization import find_chunk_boundaries


def _encode_chunk_worker(args):
    file_path, boundaries, start_idx, end_idx, tokenizer_state_path, output_file_path = args
    with open(tokenizer_state_path, 'rb') as f:
        tokenizer_state = pickle.load(f)

    tokenizer = BPETokenizer()
    tokenizer.vocabulary = tokenizer_state['vocabulary']
    tokenizer.reverseVocab = tokenizer_state['reverseVocab']
    tokenizer.merges = tokenizer_state['merges']

    del tokenizer_state
    gc.collect()

    token_count = 0
    with open(output_file_path, 'wb') as out_f:
        with open(file_path, 'rb') as in_f:
            for i in range(start_idx, end_idx):
                in_f.seek(boundaries[i])
                chunk_text = in_f.read(boundaries[i+1] - boundaries[i]).decode('utf-8', errors='ignore')
                chunk_tokens = tokenizer.encode(chunk_text)

                np_tokens = np.array(chunk_tokens, dtype=np.uint16)
                np_tokens.tofile(out_f)
                token_count += len(chunk_tokens)

                del chunk_text
                del chunk_tokens
                del np_tokens
                gc.collect()

    return (output_file_path, token_count)


def preprocess_data(
    train_file: str,
    val_file: str = None,
    vocab_size: int = 50257,
    output_dir: str = None,
    num_workers: int = 4
):
    script_path = Path(__file__).resolve()
    package_root = script_path.parent.parent

    if output_dir is None:
        output_dir = package_root / 'data/tokenized'
    else:
        output_dir = package_root / 'data' / output_dir

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Training BPE Tokenizer..\n")
    print("  Note: Your BPETokenizer caches results automatically in ./cache/")
    print()

    tokenizer = BPETokenizer()
    tokenizer.train_bpe(train_file, vocab_size=vocab_size, checkpoint_freq=10)

    print("Finished training\n")

    tokenizer_state = {
        'vocabulary': tokenizer.vocabulary,
        'reverseVocab': tokenizer.reverseVocab,
        'merges': tokenizer.merges
    }

    temp_tokenizer_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl')
    tokenizer_state_path = temp_tokenizer_file.name
    pickle.dump(tokenizer_state, temp_tokenizer_file)
    temp_tokenizer_file.close()

    print(f"  Saved tokenizer state to: {tokenizer_state_path}")

    del tokenizer
    del tokenizer_state
    gc.collect()

    print("=" * 70)
    print("Tokenizing data..")
    print("=" * 70)
    print(f"  Reading from: {train_file}")
    print(f"  Using {num_workers} parallel workers")

    desired_num_chunks = num_workers * 10  # 10 chunks per worker
    with open(train_file, 'rb') as f:
        train_boundaries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")

    num_chunks = len(train_boundaries) - 1

    if num_workers > 1:
        chunks_per_worker = num_chunks // num_workers
        remainder = num_chunks % num_workers

        worker_args = []
        worker_temp_files = []
        current_idx = 0
        for worker_id in range(num_workers):
            worker_chunk_count = chunks_per_worker + (1 if worker_id < remainder else 0)
            start_idx = current_idx
            end_idx = current_idx + worker_chunk_count

            temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=f'_worker{worker_id}.bin')
            worker_temp_path = temp_file.name
            temp_file.close()
            worker_temp_files.append(worker_temp_path)

            worker_args.append((
                train_file,
                train_boundaries,
                start_idx,
                end_idx,
                tokenizer_state_path,
                worker_temp_path
            ))
            current_idx = end_idx

        with Pool(num_workers) as pool:
            worker_file_info = []
            total_tokens = 0
            for file_path, token_count in tqdm(pool.imap(_encode_chunk_worker, worker_args),
                                               total=num_workers, desc="  Progress", unit="worker"):
                worker_file_info.append((file_path, token_count))
                total_tokens += token_count
    else:
        temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='_worker0.bin')
        worker_temp_path = temp_file.name
        temp_file.close()
        worker_temp_files = [worker_temp_path]

        file_path, token_count = _encode_chunk_worker((
            train_file,
            train_boundaries,
            0,
            num_chunks,
            tokenizer_state_path,
            worker_temp_path
        ))
        worker_file_info = [(file_path, token_count)]
        total_tokens = token_count

    print(f"  Total tokens: {total_tokens:,}")

    train_dir = output_dir / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)
    train_basename = os.path.splitext(os.path.basename(train_file))[0]
    train_output_path = train_dir / f"{train_basename}_tokens_.bin"

    train_memmap = np.memmap(train_output_path, dtype=np.uint16, mode='w+', shape=(total_tokens,))

    offset = 0
    for worker_id, (worker_file, token_count) in enumerate(worker_file_info):
        worker_memmap = np.memmap(worker_file, dtype=np.uint16, mode='r', shape=(token_count,))
        train_memmap[offset:offset + token_count] = worker_memmap
        
        offset += token_count
        del worker_memmap
        os.unlink(worker_file)
        gc.collect()

    train_memmap.flush()
    del train_memmap
    gc.collect()

    if val_file is None:
        os.unlink(tokenizer_state_path)

        print("=" * 70)
        print("Preprocessing complete!")
        print("=" * 70)
        print("\n Created files:")
        print(f"  1. {train_output_path}")
        print(f"  Training tokens: {total_tokens:,}")
        return

    print(f"\n  Reading from: {val_file}")

    with open(val_file, 'rb') as f:
        val_boundaries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")

    num_val_chunks = len(val_boundaries) - 1

    if num_workers > 1:
        chunks_per_worker = num_val_chunks // num_workers
        remainder = num_val_chunks % num_workers

        worker_args = []
        val_worker_temp_files = []
        current_idx = 0
        for worker_id in range(num_workers):
            worker_chunk_count = chunks_per_worker + (1 if worker_id < remainder else 0)
            start_idx = current_idx
            end_idx = current_idx + worker_chunk_count

            temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=f'_val_worker{worker_id}.bin')
            worker_temp_path = temp_file.name
            temp_file.close()
            val_worker_temp_files.append(worker_temp_path)

            worker_args.append((
                val_file,
                val_boundaries,
                start_idx,
                end_idx,
                tokenizer_state_path,
                worker_temp_path
            ))
            current_idx = end_idx

        with Pool(num_workers) as pool:
            val_worker_file_info = []
            total_val_tokens = 0
            for file_path, token_count in tqdm(pool.imap(_encode_chunk_worker, worker_args),
                                               total=num_workers, desc="  Progress", unit="worker"):
                val_worker_file_info.append((file_path, token_count))
                total_val_tokens += token_count
    else:
        temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='_val_worker0.bin')
        worker_temp_path = temp_file.name
        temp_file.close()
        val_worker_temp_files = [worker_temp_path]

        file_path, token_count = _encode_chunk_worker((
            val_file,
            val_boundaries,
            0,
            num_val_chunks,
            tokenizer_state_path,
            worker_temp_path
        ))
        val_worker_file_info = [(file_path, token_count)]
        total_val_tokens = token_count

    print(f"Total validation tokens: {total_val_tokens:,}")

    val_dir = output_dir / 'val'
    val_dir.mkdir(parents=True, exist_ok=True)
    val_basename = os.path.splitext(os.path.basename(val_file))[0]
    val_output_path = val_dir / f"{val_basename}_val_tokens_.bin"

    val_memmap = np.memmap(val_output_path, dtype=np.uint16, mode='w+', shape=(total_val_tokens,))

    offset = 0
    for worker_id, (worker_file, token_count) in enumerate(val_worker_file_info):
        worker_memmap = np.memmap(worker_file, dtype=np.uint16, mode='r', shape=(token_count,))
        val_memmap[offset:offset + token_count] = worker_memmap
        offset += token_count

        del worker_memmap
        os.unlink(worker_file)
        gc.collect()

    val_memmap.flush()
    del val_memmap
    gc.collect()

    os.unlink(tokenizer_state_path)
    print("\n Created files:")
    print(f"  1. {train_output_path}")
    print(f"  2. {val_output_path}")
    print("  3. Tokenizer cache (in ./cache/ directory)")
    print("\n Dataset statistics:")
    print(f"  Total training tokens: {total_tokens:,}")
    print(f"  Total validation tokens: {total_val_tokens:,}")
    print(f"  Training file size: {os.path.getsize(train_output_path) / 1024 / 1024:.2f} MB")
    print(f"  Validation file size: {os.path.getsize(val_output_path) / 1024 / 1024:.2f} MB")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess text data for training")

    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to raw training text file")
    parser.add_argument("--val_file", type=str, default=None,
                        help="Path to raw validation text file (optional)")
    parser.add_argument("--vocab_size", type=int, default=50257,
                        help="BPE vocabulary size (default: 50257 for GPT-2)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save tokenized files (default: cs336_basics/data)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers for chunked encoding (default: 4)")

    args = parser.parse_args()

    profiler = cProfile.Profile()
    profiler.enable()

    preprocess_data(
        train_file=args.train_file,
        val_file=args.val_file,
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
        num_workers=args.num_workers
    )

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(30)
    stats.sort_stats('time')
    stats.print_stats(20)
