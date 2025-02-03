# tokenizer.py
def simple_tokenize(text):
    """
    A simple tokenizer that lowercases and splits on whitespace.
    """
    # Lowercase the text.
    text = text.lower()
    # Split on whitespace.
    tokens = text.split()
    return tokens

def tokens_to_indices(tokens, vocab_dictionary, unk_index=0):
    """
    Convert a list of tokens to a list of indices using vocab_dictionary.
    If a token is not found, return unk_index.
    """
    return [vocab_dictionary.get(token, unk_index) for token in tokens]

if __name__ == "__main__":
    # Quick test
    from load_embeddings import load_pretrained_word2vec
    vocab_dictionary, _ = load_pretrained_word2vec()
    sample_text = "This is a test sentence for the two tower model."
    tokens = simple_tokenize(sample_text)
    indices = tokens_to_indices(tokens, vocab_dictionary)
    print("Tokens:", tokens)
    print("Indices:", indices)