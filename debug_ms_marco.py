# debug_ms_marco.py
from datasets import load_dataset
import pprint

def debug_ms_marco():
    # Load the dataset using the v1.1 configuration
    dataset = load_dataset("ms_marco", "v1.1")
    train_dataset = dataset["train"]
    
    print("Train dataset size:", len(train_dataset))
    print("\n--- Printing the first 5 samples for inspection ---\n")
    
    for i in range(5):
        sample = train_dataset[i]
        print(f"Sample {i}:")
        pprint.pprint(sample)
        print("\n" + "-"*80 + "\n")
    
if __name__ == "__main__":
    debug_ms_marco()