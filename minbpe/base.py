def count_pairs(tokens, counts = None):
    """
    Count the frequency of adjacent token pairs.

    Args:
        tokens (list): List of tokens to count pairs from.

    Returns:
        dict: A dictionary with token pairs as keys and their frequencies as values.
    """
    pairs = {} if counts is None else counts
    for i in range(len(tokens) - 1):
        pairs[tokens[i], tokens[i+1]] = pairs.get((tokens[i], tokens[i+1]), 0) + 1
    return pairs

def merge_tokens(tokens, pair: tuple, new_token: int):
    """
    Merge specified token pairs into a new token.

    Args:
        tokens (list): List of tokens to merge.
        pair (tuple): The pair of tokens to merge.
        new_token (int): The new token to replace the pair with.

    Returns:
        list: A new list of tokens with the specified pair merged.
    """
    output = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
            output.append(new_token)
            i += 2
        else:
            output.append(tokens[i])
        i += 1
    return output

class Tokenizer:
    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text):
        raise NotImplementedError
    
    def encode(self, text):
        raise NotImplementedError
    
    def decode(self, ids):
        raise NotImplementedError

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
    
