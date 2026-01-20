import regex as re
import pickle
import os
from pathlib import Path
from collections import Counter
import json
from tqdm import tqdm
import shutil


class BPETokenizer:
    def __init__(self, special_tokens: list[str] = []):
        self.special_tokens = special_tokens
        self.vocabulary: dict[bytes, int] = {}
        self.reverseVocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []
        self.sorted_merges = {}
        self._initialize_vocabulary()

    def _process_corpus_chunks(self, input_path: str):
        file_size = os.path.getsize(input_path)
        chunk_size = 1024 * 1024  # 1MB

        word_freqs = Counter()
        pair_freqs = Counter()

        with open(input_path, encoding="utf-8") as f:
            buffer = ""
            with tqdm(total=file_size, desc="Processing corpus", unit="B", unit_scale=True) as pbar:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        if buffer:
                            encoded_chunk = self._pre_tokenize(buffer)
                            for word in encoded_chunk:
                                word_tuple = tuple(word)
                                word_freqs[word_tuple] += 1
                                for i in range(len(word) - 1):
                                    pair_freqs[(word[i], word[i + 1])] += 1
                        break
                    buffer += chunk
                    last_newline = buffer.rfind('\n')
                    if last_newline != -1:
                        process_text = buffer[:last_newline + 1]
                        buffer = buffer[last_newline + 1:]
                    else:
                        if len(buffer) > 100:
                            process_text = buffer[:-10]
                            buffer = buffer[-10:]
                        else:
                            pbar.update(len(chunk))
                            continue

                    encoded_chunk = self._pre_tokenize(process_text)

                    for word in encoded_chunk:
                        word_tuple = tuple(word)
                        word_freqs[word_tuple] += 1

                        for i in range(len(word) - 1):
                            pair_freqs[(word[i], word[i + 1])] += 1

                    pbar.update(len(chunk))

        return word_freqs, pair_freqs

    def _get_cache_path(self, input_path: str, vocab_size: int):
        base = Path(input_path)
        tokenizer_dir = Path(__file__).parent
        cache_dir = tokenizer_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        st_tag = f"st{len(self.special_tokens or [])}"
        return cache_dir / f"{base.stem}_v{vocab_size}_{st_tag}_cache.pkl"

    def _get_checkpoint_dir(self, input_path: str):
        base = Path(input_path)
        tokenizer_dir = Path(__file__).parent
        base_dir = tokenizer_dir.parent
        checkpoint_dir = base_dir / "checkpoints" / "tokenizer" / base.stem
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir
    
    def _save(self, cache_path: Path, vocab_size: int):
        with open(cache_path, "wb") as f:
                pickle.dump({
                    "vocabulary": self.vocabulary,
                    "reverseVocab": self.reverseVocab,
                    "merges": self.merges,
                    "vocab_size": vocab_size,
                    "special_tokens": self.special_tokens,
                }, f)

    def _load(self, input_path: str, vocab_size: int):
        cache_path = self._get_cache_path(input_path, vocab_size)
        if not cache_path.exists():
            return False
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
            self.vocabulary = data["vocabulary"]
            self.reverseVocab = data["reverseVocab"]
            self.merges = data["merges"]
            self.special_tokens = data["special_tokens"]

        self.sorted_merges = {
            (a, b): i
            for i, (a, b) in enumerate(self.merges)
        }
        print(f"Loaded tokenizer from cache: {cache_path}")
        return True


    def _pre_tokenize(self, input_str: str):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if type(input_str) is not str:
            raise ValueError("Input not found")
        if not self.special_tokens:
            chunks = [input_str]
            special_set = set()
        else:
            split_pat = f"({'|'.join(map(re.escape, self.special_tokens))})"
            chunks = re.split(split_pat, input_str)
            special_set = set(self.special_tokens)

        encoded_text = []

        for chunk in chunks:
            if chunk in special_set:
                token_bytes = chunk.encode("utf-8")
                token_id = self.vocabulary[token_bytes]
                encoded_text.append([token_id])
            else:
                tokens = re.findall(PAT, chunk)
                for tok in tokens:
                    byte_values = tok.encode("utf-8")
                    token_ids = [self.vocabulary[bytes([b])] for b in byte_values]
                    encoded_text.append(token_ids)

        return encoded_text

    def _initialize_vocabulary(self):
        self.vocabulary = {bytes([x]): x for x in range(256)}
        n = len(self.vocabulary)
        for token in self.special_tokens:
            self.vocabulary[token.encode("utf-8")] = n
            n+=1
        self.reverseVocab = {v: k for k, v in self.vocabulary.items()}

    def _save_checkpoint(self, checkpoint_dir: Path, iteration: int, next_id: int, vocab_size: int):
        checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iteration}.pkl"
        manifest_path = checkpoint_dir / "manifest.json"

        with open(checkpoint_path, "wb") as f:
            pickle.dump({
                "iteration": iteration,
                "vocabulary": self.vocabulary,
                "reverseVocab": self.reverseVocab,
                "merges": self.merges,
                "next_id": next_id,
                "vocab_size": vocab_size,
                "special_tokens": self.special_tokens,
            }, f)

        manifest = {"latest_checkpoint": iteration, "checkpoint_file": checkpoint_path.name}
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        print(f"Checkpoint saved at iteration {iteration}")

    def _load_checkpoint(self, checkpoint_dir: Path):
        manifest_path = checkpoint_dir / "manifest.json"

        if not manifest_path.exists():
            return None

        with open(manifest_path) as f:
            manifest = json.load(f)

        checkpoint_path = checkpoint_dir / manifest["checkpoint_file"]
        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
        return checkpoint

    def _merge_word(self, word, merge_pair, merged_id):
        if len(word) < 2:
            return word

        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == merge_pair[0] and word[i + 1] == merge_pair[1]:
                new_word.append(merged_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def _update_pairs_after_merge(self, word_freqs, best_pair, next_id):
        new_word_freqs = {}
        pair_delta = Counter()

        for word, freq in word_freqs.items():
            has_pair = False
            i = 0
            while i < len(word) - 1:
                if word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    has_pair = True
                    break
                i += 1

            if not has_pair:
                new_word_freqs[word] = freq
                continue

            for i in range(len(word) - 1):
                pair_delta[(word[i], word[i + 1])] -= freq

            new_word = self._merge_word(word, best_pair, next_id)
            new_word_freqs[new_word] = freq

            for i in range(len(new_word) - 1):
                pair_delta[(new_word[i], new_word[i + 1])] += freq

        return new_word_freqs, pair_delta

    def _merge_bpe(self, input_enc: list):
        output = []
        for word in input_enc:
            if len(word) < 2:
                output.append(word)
                continue

            word = list(word)

            while len(word) > 1:
                best_rank = float('inf')
                best_pos = None

                for i in range(len(word) - 1):
                    left_bytes = self.reverseVocab[word[i]]
                    right_bytes = self.reverseVocab[word[i + 1]]
                    pair = (left_bytes, right_bytes)

                    if pair in self.sorted_merges:
                        rank = self.sorted_merges[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pos = i

                if best_pos is None:
                    break

                left_bytes = self.reverseVocab[word[best_pos]]
                right_bytes = self.reverseVocab[word[best_pos + 1]]
                merged_bytes = left_bytes + right_bytes
                merged_id = self.vocabulary[merged_bytes]

                word[best_pos] = merged_id
                del word[best_pos + 1]

            output.append(word)
        return output
    
    def _reconstruct_frequencies(self, input_path: str):
        file_size = os.path.getsize(input_path)
        chunk_size = 1024 * 1024  # 1MB

        word_freqs = Counter()
        pair_freqs = Counter()
        self.sorted_merges = {
            (a, b): i
            for i, (a, b) in enumerate(self.merges)
        }

        with open(input_path, encoding="utf-8") as f:
            buffer = ""
            with tqdm(total=file_size, desc="Reconstructing frequencies", unit="B", unit_scale=True) as pbar:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        if buffer:
                            encoded_chunk = self._pre_tokenize(buffer)
                            merged_chunk = self._merge_bpe(encoded_chunk)
                            for word in merged_chunk:
                                word_tuple = tuple(word)
                                word_freqs[word_tuple] += 1
                                for i in range(len(word) - 1):
                                    pair_freqs[(word[i], word[i + 1])] += 1
                        break

                    buffer += chunk
                    last_newline = buffer.rfind('\n')
                    if last_newline != -1:
                        process_text = buffer[:last_newline + 1]
                        buffer = buffer[last_newline + 1:]
                    else:
                        if len(buffer) > 100:
                            process_text = buffer[:-10]
                            buffer = buffer[-10:]
                        else:
                            pbar.update(len(chunk))
                            continue

                    encoded_chunk = self._pre_tokenize(process_text)
                    merged_chunk = self._merge_bpe(encoded_chunk)

                    for word in merged_chunk:
                        word_tuple = tuple(word)
                        word_freqs[word_tuple] += 1
                        for i in range(len(word) - 1):
                            pair_freqs[(word[i], word[i + 1])] += 1

                    pbar.update(len(chunk))

        return word_freqs, pair_freqs

    def train_bpe(self, input_path: str, vocab_size: int, checkpoint_freq: int = 100):
        if self._load(input_path, vocab_size):
            return self.reverseVocab, self.merges

        checkpoint_dir = self._get_checkpoint_dir(input_path)
        checkpoint = self._load_checkpoint(checkpoint_dir)

        if checkpoint:
            self.vocabulary = checkpoint["vocabulary"]
            self.reverseVocab = checkpoint["reverseVocab"]
            self.merges = checkpoint["merges"]
            next_id = checkpoint["next_id"]
            start_iteration = checkpoint["iteration"] + 1
            print(f"Resuming training from iteration {start_iteration}")

            word_freqs, pair_freqs = self._reconstruct_frequencies(input_path)
        else:
            print("No checkpoint found. Starting fresh training...")
            self._initialize_vocabulary()
            word_freqs, pair_freqs = self._process_corpus_chunks(input_path)

            next_id = len(self.vocabulary)
            start_iteration = 0
            self._save_checkpoint(checkpoint_dir, 0, next_id, vocab_size)

        iteration = start_iteration
        total_merges = vocab_size - len(self.vocabulary)

        with tqdm(total=total_merges, desc="Training BPE", unit=" merges", initial=iteration) as pbar:
            while len(self.vocabulary) < vocab_size:
                if not pair_freqs:
                    break

                best_pair = max(
                    pair_freqs.items(),
                    key=lambda x: (x[1], (self.reverseVocab[x[0][0]], self.reverseVocab[x[0][1]]))
                )[0]

                merged_bytes = self.reverseVocab[best_pair[0]] + self.reverseVocab[best_pair[1]]
                if merged_bytes in self.vocabulary:
                    del pair_freqs[best_pair]
                    continue

                self.vocabulary[merged_bytes] = next_id
                self.reverseVocab[next_id] = merged_bytes
                self.merges.append((self.reverseVocab[best_pair[0]], self.reverseVocab[best_pair[1]]))
                word_freqs, pair_delta = self._update_pairs_after_merge(word_freqs, best_pair, next_id)

                for pair, delta in pair_delta.items():
                    pair_freqs[pair] += delta
                    if pair_freqs[pair] <= 0:
                        del pair_freqs[pair]

                next_id += 1
                iteration += 1
                pbar.update(1)

                if iteration % checkpoint_freq == 0:
                    self._save_checkpoint(checkpoint_dir, iteration, next_id, vocab_size)

        self.sorted_merges = {
            (a, b): i
            for i, (a, b) in enumerate(self.merges)
        }

        cache_path = self._get_cache_path(input_path, vocab_size)
        self._save(cache_path, vocab_size)
        print(f"Training complete. Final vocabulary size: {len(self.vocabulary)}")

        # if checkpoint_dir.exists():
        #     shutil.rmtree(checkpoint_dir)

        return self.reverseVocab, self.merges
    
    def encode(self, input_str: str):
        tokenized_str = self._pre_tokenize(input_str)
        encoded_str = self._merge_bpe(tokenized_str)
        return [token for word in encoded_str for token in word]

    def decode(self, tokenized_input: list[int]):
        all_bytes = b"".join(self.reverseVocab[token] for token in tokenized_input)
        return all_bytes.decode("utf-8", errors="replace")