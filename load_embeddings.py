# load_embeddings.py
import gensim.downloader as api
import torch
import numpy as np

def load_pretrained_word2vec():
    print("Loading pretrained Word2Vec model (Google News 300)...")
    # This will download (if needed) and load the model
    model = api.load('word2vec-google-news-300')  
    embedding_dim = model.vector_size
    print(f"Model loaded with embedding dimension: {embedding_dim}")

    # Build vocabulary: word -> index
    vocab_list = model.index_to_key  # List of words in order of frequency
    vocab_dictionary = {word: i for i, word in enumerate(vocab_list)}
    vocab_size = len(vocab_dictionary)
    print(f"Vocabulary size: {vocab_size}")

    # Create embedding matrix (vocab_size x embedding_dim)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in vocab_dictionary.items():
        embedding_matrix[idx] = model[word]
    
    # Convert to a torch tensor.
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)
    
    return vocab_dictionary, embedding_matrix

if __name__ == "__main__":
    vocab_dictionary, embedding_matrix = load_pretrained_word2vec()
    print("Pretrained embeddings loaded successfully.")