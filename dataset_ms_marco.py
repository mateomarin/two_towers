# dataset_ms_marco.py
import torch
from torch.utils.data import Dataset
from tokenizer import simple_tokenize, tokens_to_indices

class MSMarcoTripletDataset(Dataset):
    def __init__(self, triplets, vocab_dictionary, max_length=128):
        """
        triplets: list of (query, positive, negative, weight) tuples.
        vocab_dictionary: dictionary mapping tokens to indices.
        max_length: maximum sequence length (for padding/truncation).
        """
        self.triplets = triplets
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
        query, pos, neg, weight = self.triplets[idx]
        q_tokens = simple_tokenize(query)
        p_tokens = simple_tokenize(pos)
        n_tokens = simple_tokenize(neg)
        q_indices = self.pad_sequence(tokens_to_indices(q_tokens, self.vocab_dictionary))
        p_indices = self.pad_sequence(tokens_to_indices(p_tokens, self.vocab_dictionary))
        n_indices = self.pad_sequence(tokens_to_indices(n_tokens, self.vocab_dictionary))
        return (torch.tensor(q_indices, dtype=torch.long),
                torch.tensor(p_indices, dtype=torch.long),
                torch.tensor(n_indices, dtype=torch.long),
                torch.tensor(weight, dtype=torch.float))