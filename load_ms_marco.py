# load_ms_marco.py
from datasets import load_dataset

def load_ms_marco(cache_dir="data/cache"):
    """
    Loads the MS MARCO dataset using the Hugging Face datasets library.
    We use the "v1.1" configuration (available configurations are "v1.1" and "v2.1").
    The dataset will be cached in the specified cache_dir so that subsequent calls load from cache.
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    dataset = load_dataset("ms_marco", "v1.1", cache_dir=cache_dir)
    return dataset["train"], dataset["validation"], dataset["test"]

if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_ms_marco()
    print("Train dataset length:", len(train_ds))