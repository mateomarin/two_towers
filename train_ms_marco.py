# train_ms_marco.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import json
import numpy as np
from datasets import load_dataset

from two_tower_model import TwoTowerModel
from dataset_ms_marco import MSMarcoTripletDataset
from tokenizer import simple_tokenize, tokens_to_indices

# ------------------------
# Step 1: Load MS MARCO dataset.
def load_ms_marco(cache_dir="data/cache"):
    # Using v1.1 configuration (available: v1.1, v2.1)
    dataset = load_dataset("ms_marco", "v1.1", cache_dir=cache_dir)
    return dataset["train"], dataset["validation"], dataset["test"]

print("Loading MS MARCO dataset...")
train_dataset, val_dataset, test_dataset = load_ms_marco()
print("Train dataset size:", len(train_dataset))

# ------------------------
# Step 2: Debug – Inspect a few samples to understand structure.
print("\n--- Debug: Printing first 3 samples ---")
for i in range(3):
    sample = train_dataset[i]
    print(f"Sample {i}:")
    print(sample)
    print("-" * 80)

# ------------------------
# Step 3: Build negative sampling corpus & generate triplets.
# Here we assume that each training sample has a "query" field and a "passages" field.
# The "passages" field is a dict with keys "is_selected" (a list of 0/1) and "passage_text" (a list of passages).
# We extract the first passage where is_selected == 1 as the positive passage.
all_passages = set()

def extract_positive(sample):
    pos = None
    if "passages" in sample and sample["passages"]:
        passages = sample["passages"]
        if isinstance(passages, dict):
            is_selected = passages.get("is_selected", [])
            passage_texts = passages.get("passage_text", [])
            for idx, sel in enumerate(is_selected):
                if sel == 1 and idx < len(passage_texts):
                    pos = passage_texts[idx]
                    break
    return pos

# Build the negative sampling corpus from all training samples.
for sample in train_dataset:
    pos_text = extract_positive(sample)
    if pos_text:
        all_passages.add(pos_text)
all_passages = list(all_passages)
print("Negative sampling corpus size:", len(all_passages))

def sample_negative(positive):
    # Sample a negative passage that is not identical to the positive.
    if not all_passages:
        return ""
    neg = positive
    tries = 0
    while neg == positive and tries < 10:
        neg = random.choice(all_passages)
        tries += 1
    return neg

# Generate triplets.
triplets = []
num_samples = min(10000, len(train_dataset))  # for demonstration, use up to 10k samples
for i in range(num_samples):
    sample = train_dataset[i]
    query = sample.get("query", "")
    pos = extract_positive(sample)
    if not pos or not query:
        continue  # Skip if no positive or no query.
    neg = sample_negative(pos)
    triplets.append((query, pos, neg))
print("Generated", len(triplets), "triplets for training.")

# For debugging, print a few triplets:
print("\n--- Example Triplets ---")
for t in triplets[:3]:
    print("Query:", t[0])
    print("Positive:", t[1][:150] + "...")
    print("Negative:", t[2][:150] + "...")
    print("-" * 80)

# ------------------------
# Step 4: Build or load vocabulary and embeddings.
# We load pretrained Word2Vec (Google News 300) via gensim.
import gensim.downloader as api
print("Loading pretrained Word2Vec (Google News 300)...")
w2v_model = api.load('word2vec-google-news-300')
embedding_dim = w2v_model.vector_size
# Build vocabulary from the pretrained model.
vocab_list = w2v_model.index_to_key
# Lowercase the tokens to match our simple_tokenize (which lowercases input).
vocab_dictionary = {word.lower(): idx for idx, word in enumerate(vocab_list)}
vocab_size = len(vocab_dictionary)
print("Pretrained vocabulary size:", vocab_size)

# Create embedding matrix.
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, idx in vocab_dictionary.items():
    try:
        embedding_matrix[idx] = w2v_model[word]
    except KeyError:
        embedding_matrix[idx] = np.random.randn(embedding_dim)
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)

# ------------------------
# Step 5: Create the PyTorch Dataset and DataLoader.
max_length = 128  # You can adjust as needed.
# Our MSMarcoTripletDataset (defined in dataset_ms_marco.py) expects triplets in the form (query, positive, negative).
dataset = MSMarcoTripletDataset(triplets, vocab_dictionary, max_length=max_length)
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# ------------------------
# Step 6: Instantiate the Two Tower Model.
hidden_dim = 128
model = TwoTowerModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                      pretrained_embeddings=embedding_matrix, freeze_embeddings=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ------------------------
# Step 7: Set up optimizer and loss function.
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.TripletMarginLoss(margin=0.2, p=2)

# ------------------------
# Step 8: Training loop.
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch_idx, (q, p, n) in enumerate(dataloader):
        q, p, n = q.to(device), p.to(device), n.to(device)
        optimizer.zero_grad()
        q_enc, pos_enc, neg_enc = model(q, p, n)
        loss = criterion(q_enc, pos_enc, neg_enc)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

# ------------------------
# Step 9: Save the model and embeddings.
torch.save(model.state_dict(), "ms_marco_two_tower.pt")
np.save("ms_marco_embeddings.npy", model.in_embeddings.weight.data.cpu().numpy())
with open('ms_marco_word_to_index.json', 'w') as f:
    json.dump(vocab_dictionary, f)
print("Training complete. Model and embeddings saved.")

# ------------------------
# Step 10: Sanity Check Inference.
import torch.nn.functional as F

def cache_document_encodings(model, docs, device, max_length):
    model.eval()
    encodings = []
    with torch.no_grad():
        for doc in docs:
            tokens = simple_tokenize(doc)
            indices = tokens_to_indices(tokens, vocab_dictionary)
            if len(indices) < max_length:
                indices += [0] * (max_length - len(indices))
            else:
                indices = indices[:max_length]
            doc_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
            enc = model.encode_doc(doc_tensor)
            encodings.append(F.normalize(enc, p=2, dim=1).squeeze(0))
    return torch.stack(encodings)

# Use the positive passages from the first 100 triplets as candidate documents.
candidate_docs = []
for t in triplets[:100]:
    candidate_docs.append(t[1])
cached_docs = cache_document_encodings(model, candidate_docs, device, max_length)

def infer_top_k(query_text, model, cached_docs, vocab_dictionary, max_length, k=1, device="cpu"):
    tokens = simple_tokenize(query_text)
    indices = tokens_to_indices(tokens, vocab_dictionary)
    if len(indices) < max_length:
        indices += [0] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    query_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        q_enc = model.encode_query(query_tensor)
    q_enc = F.normalize(q_enc, p=2, dim=1)
    sims = torch.matmul(q_enc, cached_docs.t()).squeeze(0)
    topk = torch.topk(sims, k=k)
    return topk.indices.cpu(), topk.values.cpu()

print("\n--- Sanity Check Inference ---")
query_text = input("Enter a query: ")
top_indices, top_scores = infer_top_k(query_text, model, cached_docs, vocab_dictionary, max_length, k=1, device=device)
best_doc = candidate_docs[top_indices[0].item()]
print(f"Best matching document (score {top_scores[0].item():.4f}):")
print(best_doc)