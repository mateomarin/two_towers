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
import torch.nn.functional as F
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

def cache_document_encodings(model, doc_token_list, device):
    """
    Pre-compute and cache the document embeddings for a list of tokenized documents.
    Each document is expected to be a list of token indices.
    """
    model.eval()
    doc_encodings = []
    with torch.no_grad():
        for tokens in doc_token_list:
            # Convert the token list to a tensor and add a batch dimension.
            doc_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            enc = model.encode_doc(doc_tensor)
            enc = F.normalize(enc, p=2, dim=1)
            doc_encodings.append(enc.squeeze(0))
    return torch.stack(doc_encodings)

def infer_top_k(model, query_text, cached_doc_encodings, vocab_dictionary, max_length=10, k=1, device="cpu"):
    """
    Encodes an input query and returns the indices and similarity scores of the top-k candidate documents.
    """
    # Tokenize the query and convert to indices.
    tokens = simple_tokenize(query_text)
    indices = tokens_to_indices(tokens, vocab_dictionary)
    # Pad or truncate the sequence to max_length.
    if len(indices) < max_length:
        indices += [0] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    query_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        query_vec = model.encode_query(query_tensor)
    query_vec = F.normalize(query_vec, p=2, dim=1)  # Ensure it’s normalized

    # Compute cosine similarities with cached document embeddings.
    # (Since vectors are normalized, the dot product is the cosine similarity.)
    similarities = torch.matmul(query_vec, cached_doc_encodings.t()).squeeze(0)
    topk = torch.topk(similarities, k=k)
    return topk.indices.cpu(), topk.values.cpu()


# -------------------
# After training is complete:
print("Training complete!")
# --- Sanity Check Inference ---

# For the sanity check, we use the positive documents from our dummy triplets as candidate answers.
# (If you have a larger corpus, you would use that.)
dummy_triplets = [
    ("what is ai", "artificial intelligence is the simulation of human intelligence", "the capital of france is paris"),
    ("how to cook pasta", "cooking pasta involves boiling water and adding salt", "the stock market closed higher today"),
    ("what is machine learning", "machine learning is a branch of artificial intelligence", "cats are cute"),
    ("explain photosynthesis", "photosynthesis converts light energy into chemical energy", "computers process data")
]
# Extract the positive documents.
candidate_docs = [pos for (_, pos, _) in dummy_triplets]

# Convert each candidate document into token indices using our tokenizer.
max_length = 10  # Must match what is used in your dataset.
candidate_tokens = []
for doc in candidate_docs:
    tokens = simple_tokenize(doc)
    indices = tokens_to_indices(tokens, vocab_dictionary)  # vocab_dictionary was loaded in train_two_tower.py from load_embeddings
    if len(indices) < max_length:
        indices += [0] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    candidate_tokens.append(indices)

# Cache the document embeddings.
cached_docs = cache_document_encodings(model, candidate_tokens, device)

# Now, run an interactive query:
print("\n--- Sanity Check: Query Inference ---")
query_text = input("Enter a query: ")
top_indices, top_scores = infer_top_k(model, query_text, cached_docs, vocab_dictionary, max_length=max_length, k=1, device=device)
# Retrieve the best matching document from candidate_docs.
best_doc_index = top_indices[0].item()
print(f"Best match (similarity {top_scores[0].item():.4f}):")
print(candidate_docs[best_doc_index])