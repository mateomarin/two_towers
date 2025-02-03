# train_two_tower.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dummy_dataset import DummyTripletDataset
from two_tower_model import TwoTowerModel
from load_embeddings import load_pretrained_word2vec
from tokenizer import simple_tokenize, tokens_to_indices

# Step 1: Load pretrained embeddings and vocabulary
vocab_dictionary, embedding_matrix = load_pretrained_word2vec()
vocab_size = len(vocab_dictionary)
embedding_dim = embedding_matrix.shape[1]

# Step 2: Create a dummy dataset of triplets.
dummy_triplets = [
    ("what is ai", "artificial intelligence is the simulation of human intelligence", "the capital of france is paris"),
    ("how to cook pasta", "cooking pasta involves boiling water and adding salt", "the stock market closed higher today"),
    ("what is machine learning", "machine learning is a branch of artificial intelligence", "cats are cute"),
    ("explain photosynthesis", "photosynthesis converts light energy into chemical energy", "computers process data")
]
dataset = DummyTripletDataset(dummy_triplets, simple_tokenize, vocab_dictionary, max_length=10)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 3: Instantiate 2 tower model
hidden_dim = 128
model = TwoTowerModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, pretrained_embeddings=embedding_matrix, freeze_embeddings=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 4: Set up optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.TripletMarginLoss(margin=0.2, p=2)

# Step 5: Training loop
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0.0
    for query, pos, neg in dataloader:
        query, pos, neg = query.to(device), pos.to(device), neg.to(device)
        optimizer.zero_grad()
        query_emb, pos_emb, neg_emb = model(query, pos, neg)
        loss = criterion(query_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

print("Training complete!")