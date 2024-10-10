import regex as re
from .base import Tokenizer, count_pairs, merge_tokens

# GPT text split patterns from https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):
    """
    A tokenizer that uses regular expressions to split text into tokens.
    
    This tokenizer extends the base Tokenizer class and implements a regex-based
    approach for tokenization, with support for special tokens and customizable
    split patterns.
    """

    def __init__(self, pattern=None):
        """
        Initialize the RegexTokenizer.

        Args:
            pattern (str, optional): A custom regex pattern for text splitting.
                If None, the GPT-4 split pattern is used as default.
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
    
    def train(self, text, vocab_size, verbose=False):
        """
        Train the tokenizer on the given text to create a vocabulary of specified size.

        This method implements the Byte Pair Encoding (BPE) algorithm to iteratively
        merge the most frequent pairs of tokens until the desired vocabulary size
        is reached.

        Args:
            text (str): The text to train on.
            vocab_size (int): The desired size of the vocabulary.
            verbose (bool, optional): Whether to print progress. Defaults to False.
        """
        assert vocab_size >= 256
        
        num_merges = vocab_size - 256
        chunks = [list(chunk.encode('utf-8')) for chunk in re.findall(self.compiled_pattern, text)]
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        for i in range(vocab_size - 256):
            counts = {}
            for chunk in chunks:
                count_pairs(chunk, counts)

            best_pair = max(counts, key=counts.get)
            new_token = i + 256
            chunks = [merge_tokens(chunk, best_pair, new_token) for chunk in chunks]
            merges[best_pair] = new_token
            vocab[new_token] = vocab[best_pair[0]] + vocab[best_pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {best_pair} -> {new_token} ({vocab[new_token]}) had {counts[best_pair]} occurrences")

        self.vocab = vocab
        self.merges = merges
    
    def decode(self, tokens):
        """
        Decode the given list of token ids back into text.

        This method handles both regular tokens and special tokens, converting
        them back into their original string representation.

        Args:
            tokens (list): A list of token ids (integers) to decode.

        Returns:
            str: The decoded text.

        Raises:
            ValueError: If an unknown token is encountered.
        """
        vocab_items = []
        for token in tokens:
            if token in self.inverse_special_tokens:
                vocab_items.append(self.inverse_special_tokens[token].encode('utf-8'))
            elif token in self.vocab:
                vocab_items.append(self.vocab[token])
            else:
                raise ValueError(f"Unknown token: {token}")
        return b''.join(vocab_items).decode('utf-8', errors='replace')
    
    def _encode_chunk(self, text_bytes):
        """
        Encode a chunk of text bytes into token ids.

        This method applies the trained merges to convert byte sequences into token ids.

        Args:
            text_bytes (bytes): The chunk of text to encode.

        Returns:
            list: A list of token ids.
        """
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = count_pairs(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge_tokens(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """
        Encode text without considering special tokens.

        This method splits the text using the regex pattern and encodes each chunk separately.

        Args:
            text (str): The text to encode.

        Returns:
            list: A list of token ids.
        """
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Encode text, handling special tokens based on the allowed_special parameter.

        This method provides flexibility in handling special tokens during encoding,
        allowing for different behaviors based on the user's preference.

        Args:
            text (str): The text to encode.
            allowed_special (str or set): Specifies how to handle special tokens.
                Can be "all", "none", "none_raise", or a custom set of special tokens.

        Returns:
            list: A list of token ids.

        Raises:
            ValueError: If allowed_special is not recognized or if special tokens
                        are encountered when not allowed.
        """
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

        if not special:
            return self.encode_ordinary(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids