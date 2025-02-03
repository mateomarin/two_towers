# test_vocabulary.py
from word2vec.vocabulary import Vocabulary  # updated import since vocabulary.py is in the word2vec folder

# Specify the path to your text file (e.g., "data/text8")
vocab_file = "data/text8"

# Build the vocabulary. Adjust min_count if needed.
vocab = Vocabulary(vocab_file, min_count=50, subsample_t=1e-5)

# Test tokenization on a sample sentence.
sample_text = "This is a test sentence for the two tower model."
tokens = vocab.tokenize(sample_text)
token_indices = [vocab.dictionary.get(token, 0) for token in tokens]

print("Tokens:", tokens)
print("Token indices:", token_indices)