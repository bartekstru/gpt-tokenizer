# BasicTokenizer: Implements a simple Byte Pair Encoding (BPE) tokenizer
class BasicTokenizer:
    """
    Implements a simple Byte Pair Encoding (BPE) tokenizer.
    This tokenizer can be trained on a given text to create a vocabulary,
    and then used to encode and decode text using that vocabulary.
    """

    def __init__(self):
        # Initialize empty dictionaries for merges and vocabulary
        self.merges = {}
        self.vocab = {}

    def _count_pairs(self, tokens):
        """
        Count the frequency of adjacent token pairs.

        Args:
            tokens (list): List of tokens to count pairs from.

        Returns:
            dict: A dictionary with token pairs as keys and their frequencies as values.
        """
        pairs = {}
        for i in range(len(tokens) - 1):
            pairs[tokens[i], tokens[i+1]] = pairs.get((tokens[i], tokens[i+1]), 0) + 1
        return pairs
    
    def _merge_tokens(self, tokens, pair: tuple, new_token: int):
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
    
    def train(self, text, vocab_size, verbose=False):
        """
        Train the tokenizer on the given text to create a vocabulary of specified size.

        Args:
            text (str): The text to train on.
            vocab_size (int): The desired size of the vocabulary.
            verbose (bool, optional): Whether to print progress. Defaults to False.
        """
        tokens = list(text.encode('utf-8'))
        num_merges = vocab_size - 256
        merges = {}
        for i in range(num_merges):
            pair_freqs = self._count_pairs(tokens)
            best_pair = max(pair_freqs, key=pair_freqs.get)
            new_token = i + 256
            tokens = self._merge_tokens(tokens, best_pair, new_token)
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
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            stats = self._count_pairs(tokens)
            best_pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if best_pair not in self.merges:
                break
            idx = self.merges[best_pair]
            tokens = self._merge_tokens(tokens, best_pair, idx)
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

with open('taylorswift.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Example usage
tokenizer = BasicTokenizer()
tokenizer.train(text, 1000)  # Adjust vocab_size as needed

print(tokenizer.encode("Hello world"))