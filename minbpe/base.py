import unicodedata


class Tokenizer:
    """
    Base class for tokenizers.

    This class provides a foundation for implementing various tokenization strategies.
    It includes methods for training, encoding, decoding, and managing the vocabulary.

    Attributes:
        merges (dict): Dictionary of merge rules, mapping token pairs to new tokens.
        pattern (str): Regular expression pattern for tokenization.
        special_tokens (dict): Dictionary of special tokens and their corresponding IDs.
        vocab (dict): Dictionary mapping token IDs to their byte representations.
    """

    def __init__(self):
        """
        Initialize the Tokenizer with default values.

        Sets up empty merges, pattern, and special_tokens dictionaries,
        and initializes the vocabulary with basic byte representations.
        """
        # Default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {}  # (int, int) -> int
        self.pattern = ""  # str
        self.special_tokens = {}  # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab()  # int -> bytes

    def train(self, text):
        """
        Train the tokenizer on the given text.

        This method should be implemented by subclasses to define
        the specific training algorithm for the tokenizer.

        Args:
            text (str): The text to train on.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def encode(self, text):
        """
        Encode the given text into token IDs.

        This method should be implemented by subclasses to define
        the specific encoding algorithm for the tokenizer.

        Args:
            text (str): The text to encode.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def decode(self, ids):
        """
        Decode the given token IDs back into text.

        This method should be implemented by subclasses to define
        the specific decoding algorithm for the tokenizer.

        Args:
            ids (list): The list of token IDs to decode.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def _build_vocab(self):
        """
        Build the vocabulary based on merge rules and special tokens.

        This method constructs the vocabulary by combining basic byte representations,
        merge rules, and special tokens.

        Returns:
            dict: A dictionary mapping token IDs to their byte representations.
        """
        # Vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Save the tokenizer model to files.

        This method saves the tokenizer's configuration, including version,
        pattern, special tokens, and merges, to a .model file. It also saves
        a human-readable vocabulary to a .vocab file.

        Args:
            file_prefix (str): The prefix for the output files.
        """
        # Write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # Write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # Write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # The merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        # Write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # Note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # This also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # Find the children of this token, if any
                if idx in inverted_merges:
                    # If this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # Otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """
        Load a tokenizer model from a file.

        This method reads a .model file to reconstruct the tokenizer's
        configuration, including pattern, special tokens, and merges.

        Args:
            model_file (str): The path to the .model file to load.

        Raises:
            AssertionError: If the model file doesn't have the correct extension
                            or version.
        """
        assert model_file.endswith(".model")
        # Read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # Read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # Read the pattern
            self.pattern = f.readline().strip()
            # Read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # Read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()


def count_pairs(tokens, counts=None):
    """
    Count the frequency of adjacent token pairs.

    Args:
        tokens (list): List of tokens to count pairs from.
        counts (dict, optional): Existing counts dictionary to update. Defaults to None.

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


def replace_control_characters(s: str) -> str:
    """
    Replace control characters in the given string with their escaped hexadecimal representation.

    Args:
        s (str): The string to process.

    Returns:
        str: The processed string with control characters escaped.
    """
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)  # This character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}")  # Escape
    return "".join(chars)


def render_token(t: bytes) -> str:
    """
    Pretty print a token, escaping control characters.

    Args:
        t (bytes): The token to print.

    Returns:
        str: The pretty printed token.
    """
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s