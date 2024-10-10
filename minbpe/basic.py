# BasicTokenizer: Implements a simple Byte Pair Encoding (BPE) tokenizer
from .base import Tokenizer, count_pairs, merge_tokens

class BasicTokenizer(Tokenizer):
    """
    Implements a simple Byte Pair Encoding (BPE) tokenizer.
    This tokenizer can be trained on a given text to create a vocabulary,
    and then used to encode and decode text using that vocabulary.
    """

    def __init__(self):
        super().__init__()

    
    def train(self, text, vocab_size, verbose=False):
        """
        Train the tokenizer on the given text to create a vocabulary of specified size.

        Args:
            text (str): The text to train on.
            vocab_size (int): The desired size of the vocabulary.
            verbose (bool, optional): Whether to print progress. Defaults to False.
        """
        assert vocab_size > 256, "Vocab size must be greater than 256"
        tokens = list(text.encode('utf-8'))
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(vocab_size - 256):
            pair_freqs = count_pairs(tokens)
            best_pair = max(pair_freqs, key=pair_freqs.get)
            new_token = i + 256
            tokens = merge_tokens(tokens, best_pair, new_token)
            merges[best_pair] = new_token
            vocab[new_token] = vocab[best_pair[0]] + vocab[best_pair[1]]

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
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            stats = count_pairs(tokens)
            best_pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if best_pair not in self.merges:
                break
            idx = self.merges[best_pair]
            tokens = merge_tokens(tokens, best_pair, idx)
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