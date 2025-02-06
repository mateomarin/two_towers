import os
import pickle
import gensim.downloader as api
from datasets import load_dataset

def load_or_cache_word2vec(cache_dir='cache'):
    """Load Word2Vec embeddings from cache or download if not cached"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'word2vec.pkl')
    
    if os.path.exists(cache_path):
        print("Loading Word2Vec embeddings from cache...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print("Downloading Word2Vec embeddings...")
    word2vec = api.load('word2vec-google-news-300')
    
    print("Caching Word2Vec embeddings...")
    with open(cache_path, 'wb') as f:
        pickle.dump(word2vec, f)
    
    return word2vec

def load_or_cache_msmarco(split='train', cache_dir='cache'):
    """Load MS MARCO dataset from cache or download if not cached"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'msmarco_{split}.pkl')
    
    if os.path.exists(cache_path):
        print(f"Loading MS MARCO {split} set from cache...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print(f"Downloading MS MARCO {split} set...")
    dataset = load_dataset("ms_marco", "v1.1")
    data = dataset[split]
    
    print(f"Caching MS MARCO {split} set...")
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    
    return data 