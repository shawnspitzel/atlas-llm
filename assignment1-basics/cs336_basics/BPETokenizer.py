import regex as re
import warnings
from pprint import pprint
import pickle
from pathlib import Path
import copy

class BPETokenizer:
    def __init__(self, special_tokens: list[str] = []):
        self.special_tokens = special_tokens
        self.vocabulary: dict[bytes, int] = {}
        self.reverseVocab: dict[int, bytes] = {}
        self.corpus: None | str = None
        self.merges: list[tuple[bytes, bytes]] = []
        self.tokenizedCorpus: list = []
        self.sorted_merges = {}
        self._initialize_vocabulary()

    def _initialize_training(self, input_path: str, vocab_size:int):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self._read_corpus()
        self._pre_tokenize_corpus()

    def _read_corpus(self):
        with open(self.input_path, "r", encoding="utf-8") as f:
            self.corpus = f.read()

    def _get_cache_path(self, input_path, vocab_size):
        base = Path(input_path)
        cache_dir = Path("./cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        st_tag = f"st{len(self.special_tokens or [])}"
        return cache_dir / f"{base.stem}_v{vocab_size}_{st_tag}_cache.pkl"
    
    def _save(self, cache_path: Path):
        with open(cache_path, "wb") as f:
                pickle.dump({
                    "vocabulary": self.vocabulary,
                    "reverseVocab": self.reverseVocab,
                    "merges": self.merges,
                    "vocab_size": self.vocab_size,
                    "special_tokens": self.special_tokens,
                }, f)

    def _load(self, cache_path: Path):
        if not cache_path.exists():
            return False
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
            self.vocabulary = data["vocabulary"]
            self.reverseVocab = data["reverseVocab"]
            self.merges = data["merges"]
            self.vocab_size = data["vocab_size"]
            self.special_tokens = data["special_tokens"]

            self.tokenizedCorpus = []
            self.corpus = None
        self.sorted_merges = {
            (a, b): i
            for i, (a, b) in enumerate(self.merges)
        }
        return True

    def _pre_tokenize_corpus(self):
        self.tokenizedCorpus = self._pre_tokenize(self.corpus)

    def _pre_tokenize(self, input_str: str):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if type(input_str) != str:
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
                continue

            tokens = re.findall(PAT, chunk)
            for tok in tokens:
                encoded = list(tok.encode("utf-8"))
                encoded_text.append(encoded)

        return encoded_text

    def _initialize_vocabulary(self):
        self.vocabulary = {bytes([x]): x for x in range(256)}
        n = len(self.vocabulary)
        for token in self.special_tokens:
            self.vocabulary[token.encode("utf-8")] = n
            n+=1
        self.reverseVocab = {v: k for k, v in self.vocabulary.items()}

    def _rebuild_corpus(self, merge_pair):
        new_corpus = []
        for word in self.tokenizedCorpus:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1:
                    pair = (word[i], word[i+1])
                    if pair == merge_pair:
                        byte_string = self.reverseVocab[pair[0]] + self.reverseVocab[pair[1]]
                        new_word.append(self.vocabulary[byte_string])
                        i += 2
                        continue
                new_word.append(word[i])
                i += 1
            new_corpus.append(new_word)
        self.tokenizedCorpus = new_corpus

    def _merge_bpe(self, input_enc: list):
        output = []
        for word in input_enc:
            word = word[:]
            while True:
                best_merge_rank = None
                best_merge_index = None
                best_merge_pair = None

                for i in range(len(word) - 1):
                    left_id = word[i]
                    right_id = word[i + 1]

                    left_bytes = self.reverseVocab[left_id]
                    right_bytes = self.reverseVocab[right_id]
                    pair = (left_bytes, right_bytes)

                    if pair not in self.sorted_merges:
                        continue

                    rank = self.sorted_merges[pair]

                    if best_merge_rank is None or rank < best_merge_rank:
                        best_merge_rank = rank
                        best_merge_index = i
                        best_merge_pair = pair

                if best_merge_pair is None:
                    break
                merged_bytes = best_merge_pair[0] + best_merge_pair[1]
                merged_pair = self.vocabulary[merged_bytes]

                word = (
                    word[:best_merge_index]
                    + [merged_pair]
                    + word[best_merge_index + 2 :]
                )

            output.append(word)
        return output
    def train_bpe(self, input_path: str, vocab_size:int):
        cache_path = self._get_cache_path(input_path, vocab_size)

        self.input_path = input_path
        self.vocab_size = vocab_size

        if self._load(cache_path):
            print(f"Loaded tokenizer from cache: {cache_path}")
            return self.reverseVocab, self.merges
        
        self._initialize_training(input_path, vocab_size)

        next_id = len(self.vocabulary)  
        while True:
            freq = {}
            for word in self.tokenizedCorpus:
                for index1, index2 in zip(word, word[1:]):
                    pair = (index1, index2)
                    freq[pair] = freq.get(pair, 0) + 1
            candidates = (
                (pair, count)
                for pair, count in freq.items()
                if (self.reverseVocab[pair[0]] + self.reverseVocab[pair[1]]) not in self.vocabulary
            )
            if not candidates:
                break
            try:
                max_freq = max(candidates, key=lambda x: x[1])
            except ValueError:
                break
            merge_pair = max_freq[0]
            new_index = self.reverseVocab[merge_pair[0]] + self.reverseVocab[merge_pair[1]]
            if len(self.vocabulary) >= self.vocab_size:
                warnings.warn(f"Tokenizer vocabulary exceeded maximum length of {self.vocab_size} ",UserWarning)
                break
            self.vocabulary[new_index] = next_id
            self.reverseVocab[next_id] = new_index
            self.merges.append((self.reverseVocab[merge_pair[0]], self.reverseVocab[merge_pair[1]]))
            next_id += 1
            self._rebuild_corpus(max_freq[0])
        self._save(cache_path)
        return self.reverseVocab, self.merges
    
    def encode(self, input_str: str):
        tokenized_str = self._pre_tokenize(input_str)
        encoded_str = self._merge_bpe(tokenized_str)
        return encoded_str
    
    def decode(self, tokenized_input: list[int]):
        output = ""
        for word in tokenized_input:
            for token in word:
                output += self.reverseVocab[token].decode("utf-8")
        return output