# dummy_dataset.py
import torch

class DummyTripletDataset(torch.utils.data.Dataset):
    def __init__(self, triplets, tokenizer_fn, vocab_dictionary, max_length=20):
        """
        triplets: list of (query, positive, negative) tuples.
        tokenizer_fn: function to tokenize a string.
        vocab_dictionary: pretrained vocabulary mapping.
        max_length: maximum sequence length (pad/truncate to this length).
        """
        self.triplets = triplets
        self.tokenizer_fn = tokenizer_fn
        self.vocab_dictionary = vocab_dictionary
        self.max_length = max_length

    def pad_sequence(self, indices):
        if len(indices) < self.max_length:
            return indices + [0] * (self.max_length - len(indices))
        else:
            return indices[:self.max_length]

    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        query, pos, neg = self.triplets[idx]
        from tokenizer import simple_tokenize, tokens_to_indices
        q_tokens = simple_tokenize(query)
        p_tokens = simple_tokenize(pos)
        n_tokens = simple_tokenize(neg)

        q_indices = tokens_to_indices(q_tokens, self.vocab_dictionary)
        p_indices = tokens_to_indices(p_tokens, self.vocab_dictionary)
        n_indices = tokens_to_indices(n_tokens, self.vocab_dictionary)

        return (
            torch.tensor(self.pad_sequence(q_indices), dtype=torch.long),
            torch.tensor(self.pad_sequence(p_indices), dtype=torch.long),
            torch.tensor(self.pad_sequence(n_indices), dtype=torch.long)
        )
    
if __name__ == "__main__":
    # Create small dummy dataset
    dummy_triplets = [
        ("what is ai", "artificial intelligence is the simulation of human intelligence", "the capital of france is paris"),
        ("how to cook pasta", "cooking pasta involves boiling water and adding salt", "the stock market closed higher today")
    ]

    from load_embeddings import load_pretrained_word2vec
    vocab_dictionary, _ = load_pretrained_word2vec()
    dataset = DummyTripletDataset(
        dummy_triplets,
        None,
        vocab_dictionary=vocab_dictionary,
        max_length=20
    )
    for item in dataset:
        print(item)
