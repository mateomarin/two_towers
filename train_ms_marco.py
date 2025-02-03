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
# Function to load MS MARCO dataset (v1.1) with caching.
def load_ms_marco(cache_dir="data/cache"):
    dataset = load_dataset("ms_marco", "v1.1", cache_dir=cache_dir)
    return dataset["train"], dataset["validation"], dataset["test"]

# Debug function to inspect a few samples.
def debug_samples(train_dataset, num_samples=3):
    print("\n--- Debug: Printing first {} samples ---".format(num_samples))
    for i in range(num_samples):
        sample = train_dataset[i]
        print(f"Sample {i}:")
        print(sample)
        print("-" * 80)

# Extract all positive passages from a sample.
def extract_positives(sample):
    positives = []
    # For MS MARCO v1.1, the sample has a "passages" field (a dict with keys "is_selected" and "passage_text").
    if "passages" in sample and sample["passages"]:
        passages = sample["passages"]
        is_selected = passages.get("is_selected", [])
        passage_texts = passages.get("passage_text", [])
        for idx, text in enumerate(passage_texts):
            # If is_selected == 1, assign a higher weight (e.g., 2.0), otherwise 1.0.
            weight = 2.0 if idx < len(is_selected) and is_selected[idx] == 1 else 1.0
            positives.append((text, weight))
    return positives

# Build a negative sampling corpus from all training samples.
def build_negative_corpus(train_dataset):
    corpus = set()
    for sample in train_dataset:
        pos_list = extract_positives(sample)
        for pos, _ in pos_list:
            corpus.add(pos)
    return list(corpus)

# Sample a negative passage (ensuring it isn’t identical to the positive).
def sample_negative(positive, corpus):
    if not corpus:
        return ""
    neg = positive
    tries = 0
    while neg == positive and tries < 10:
        neg = random.choice(corpus)
        tries += 1
    return neg

# Generate triplets from the training dataset.
def generate_triplets(train_dataset, max_samples=50000):
    triplets = []
    num_samples = min(max_samples, len(train_dataset))
    for i in range(num_samples):
        sample = train_dataset[i]
        query = sample.get("query", "")
        positives = extract_positives(sample)
        if not query or not positives:
            continue
        for pos, weight in positives:
            triplets.append((query, pos, None, weight))  # Negative will be filled later.
    return triplets

# Assign negatives to the triplets.
def assign_negatives(triplets, negative_corpus):
    new_triplets = []
    for (query, pos, _, weight) in triplets:
        neg = sample_negative(pos, negative_corpus)
        new_triplets.append((query, pos, neg, weight))
    return new_triplets

# Load pretrained Word2Vec embeddings via gensim and reindex the vocabulary.
def load_pretrained_embeddings():
    import gensim.downloader as api
    print("Loading pretrained Word2Vec (Google News 300)...")
    w2v_model = api.load('word2vec-google-news-300')
    embedding_dim = w2v_model.vector_size
    vocab_list = w2v_model.index_to_key  # Original gensim vocabulary.

    # Build a mapping from lowercased token to its original token (first occurrence).
    temp_dict = {}
    for word in vocab_list:
        lw = word.lower()
        if lw not in temp_dict:
            temp_dict[lw] = word

    # Reindex so that tokens get contiguous indices.
    new_vocab_dictionary = {}
    embedding_vectors = []
    new_index = 0
    for lw, orig in temp_dict.items():
        new_vocab_dictionary[lw] = new_index
        vec = np.array(w2v_model[orig], dtype=np.float32)
        embedding_vectors.append(vec)
        new_index += 1

    embedding_matrix = np.stack(embedding_vectors, axis=0)
    vocab_dictionary = new_vocab_dictionary
    vocab_size = len(vocab_dictionary)
    print("Pretrained vocabulary size:", vocab_size)
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)
    return vocab_dictionary, embedding_matrix

# Cache document embeddings from a list of candidate documents.
def cache_document_encodings(model, docs, device, max_length):
    import torch.nn.functional as F
    encodings = []
    model.eval()
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

# Infer the top-k matching document given a query.
def infer_top_k(query_text, model, cached_docs, vocab_dictionary, max_length, k=1, device="cpu"):
    import torch.nn.functional as F
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

def main():
    # Step 1: Load MS MARCO dataset.
    print("Loading MS MARCO dataset...")
    train_dataset, val_dataset, test_dataset = load_ms_marco()
    print("Train dataset size:", len(train_dataset))

    # Step 2: Debug - Print first 3 samples.
    debug_samples(train_dataset, num_samples=3)

    # Step 3: Build negative sampling corpus and generate triplets.
    negative_corpus = build_negative_corpus(train_dataset)
    print("Negative sampling corpus size:", len(negative_corpus))
    triplets = generate_triplets(train_dataset, max_samples=50000)
    triplets = assign_negatives(triplets, negative_corpus)
    print("Generated", len(triplets), "triplets for training.")
    print("\n--- Example Triplets ---")
    for t in triplets[:3]:
        print("Query:", t[0])
        print("Positive:", t[1][:150] + "...")
        print("Negative:", t[2][:150] + "...")
        print("Weight:", t[3])
        print("-" * 80)

    # Step 4: Load pretrained embeddings and build vocabulary.
    global vocab_dictionary  # Make it global for helper functions.
    vocab_dictionary, embedding_matrix = load_pretrained_embeddings()

    # Step 5: Create the PyTorch Dataset and DataLoader.
    max_length = 128
    dataset_obj = MSMarcoTripletDataset(triplets, vocab_dictionary, max_length=max_length)
    batch_size = 64
    dataloader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=True, num_workers=0)

    # Step 6: Instantiate the Two Tower Model.
    hidden_dim = 128
    # Allow fine-tuning by setting freeze_embeddings=False.
    model = TwoTowerModel(vocab_size=len(vocab_dictionary),
                          embedding_dim=embedding_matrix.shape[1],
                          hidden_dim=hidden_dim,
                          pretrained_embeddings=embedding_matrix,
                          freeze_embeddings=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Step 7: Set up optimizer and loss function.
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Use reduction='none' to weight individual sample losses.
    criterion = nn.TripletMarginLoss(margin=0.2, p=2, reduction='none')

    # Step 8: Training loop.
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (q, p, n, w) in enumerate(dataloader):
            q, p, n, w = q.to(device), p.to(device), n.to(device), w.to(device)
            optimizer.zero_grad()
            q_enc, pos_enc, neg_enc = model(q, p, n)
            # Compute per-sample loss.
            losses = criterion(q_enc, pos_enc, neg_enc)  # shape: [batch]
            weighted_loss = (losses * w).mean()
            weighted_loss.backward()
            optimizer.step()
            total_loss += weighted_loss.item()
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {weighted_loss.item():.4f}")
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

    # Step 9: Save model and embeddings.
    torch.save(model.state_dict(), "output/ms_marco_two_tower.pt")
    np.save("output/ms_marco_embeddings.npy", model.embedding.weight.data.cpu().numpy())
    with open('output/ms_marco_word_to_index.json', 'w') as f:
        json.dump(vocab_dictionary, f)
    print("Training complete. Model and embeddings saved.")

    # Step 10: Sanity Check Inference.
    candidate_docs = [t[1] for t in triplets[:100]]
    cached_docs = cache_document_encodings(model, candidate_docs, device, max_length)
    print("\n--- Sanity Check Inference ---")
    query_text = input("Enter a query: ")
    top_indices, top_scores = infer_top_k(query_text, model, cached_docs, vocab_dictionary, max_length, k=1, device=device)
    best_doc = candidate_docs[top_indices[0].item()]
    print(f"Best matching document (score {top_scores[0].item():.4f}):")
    print(best_doc)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()