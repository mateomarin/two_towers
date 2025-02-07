import gensim.downloader as api
from pathlib import Path

def load_or_cache_word2vec(cache_dir: str = "/app/cache"):
    """
    Load Word2Vec embeddings from cache or download if not available
    """
    cache_path = Path(cache_dir) / "word2vec-google-news-300.model"
    cache_path.parent.mkdir(exist_ok=True)

    if cache_path.exists():
        print("Loading cached word2vec model...")
        return api.load('word2vec-google-news-300', cache_path)
    
    print("Downloading word2vec model...")
    model = api.load('word2vec-google-news-300')
    
    # Cache the model
    print("Caching word2vec model...")
    model.save(str(cache_path))
    
    return model 