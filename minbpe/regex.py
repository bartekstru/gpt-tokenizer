import regex as re
from .base import Tokenizer, count_pairs, merge_tokens


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
    
    def train(self, text, vocab_size, verbose=False):
        """
        Train the tokenizer on the given text to create a vocabulary of specified size.

        Args:
            text (str): The text to train on.
            vocab_size (int): The desired size of the vocabulary.
            verbose (bool, optional): Whether to print progress. Defaults to False.
        """
        

        assert vocab_size >= 256
        
        num_merges = vocab_size - 256

        chunks = re.findall(self.compiled_pattern, text)

        chunks = [list(chunk.encode('utf-8')) for chunk in chunks]

        merges = {}
        counts = {}

        for i in range(num_merges):

            for chunk in chunks:
                count_pairs(chunk, counts)

            best_pair = max(counts, key=counts.get)
            new_token = i + 256
            tokens = merge_tokens(tokens, best_pair, new_token)
            merges[best_pair] = new_token

        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        self.vocab = vocab
        self.merges = merges
    
    def encode(self, text):
        """
        Encode the given text using the trained vocabulary.

        Args:
            text (str): The text to encode.

        Returns:
            list: A list of token ids representing the encoded text.
        """
        chunks = re.findall(self.compiled_pattern, text)
        chunks = [list(chunk.encode('utf-8')) for chunk in chunks]
        tokens = []
        for chunk in chunks:
            while len(chunk) >= 2:
                stats = self._count_pairs(chunk)
                best_pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if best_pair not in self.merges:
                    break
                idx = self.merges[best_pair]
                chunk = self._merge_tokens(chunk, best_pair, idx)
            tokens.extend(chunk)
        return tokens
    
    def decode(self, ids):
        """
        Decode the given list of token ids back into text.

        Args:
            ids (list): A list of token ids to decode.

        Returns:
            str: The decoded text.
        """
        tokens = b''.join([self.vocab[id] for id in ids])
        return tokens.decode('utf-8', errors='replace')