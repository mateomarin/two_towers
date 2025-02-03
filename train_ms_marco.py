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

##########################
# Helper Functions
##########################

def load_ms_marco(cache_dir="data/cache"):
    dataset = load_dataset("ms_marco", "v1.1", cache_dir=cache_dir)
    return dataset["train"], dataset["validation"], dataset["test"]

def debug_samples(train_dataset, num_samples=3):
    print("\n--- Debug: Printing first {} samples ---".format(num_samples))
    for i in range(num_samples):
        sample = train_dataset[i]
        print(f"Sample {i}:")
        print(sample)
        print("-" * 80)

def extract_positives(sample):
    """Return a list of (positive_text, weight) tuples from a sample."""
    positives = []
    if "passages" in sample and sample["passages"]:
        passages = sample["passages"]
        is_selected = passages.get("is_selected", [])
        passage_texts = passages.get("passage_text", [])
        for idx, text in enumerate(passage_texts):
            # Higher weight for selected passages (is_selected==1), lower weight otherwise.
            weight = 2.0 if idx < len(is_selected) and is_selected[idx] == 1 else 1.0
            positives.append((text, weight))
    return positives

def build_negative_corpus(dataset):
    corpus = set()
    for sample in dataset:
        for pos, _ in extract_positives(sample):
            corpus.add(pos)
    return list(corpus)

def sample_hard_negative(query, positive, corpus, vocab_dictionary, embedding_matrix, candidate_pool_size=10):
    # Compute a simple average embedding (using pretrained embedding_matrix) for the text.
    def average_embedding(text):
        tokens = simple_tokenize(text)
        indices = tokens_to_indices(tokens, vocab_dictionary)
        if len(indices) == 0:
            return None
        vecs = embedding_matrix[indices]  # shape: [num_tokens, emb_dim]
        avg = vecs.mean(dim=0)
        norm = avg.norm() + 1e-8
        return avg / norm

    q_emb = average_embedding(query)
    if q_emb is None:
        return random.choice(corpus)
    candidates = random.sample(corpus, min(candidate_pool_size, len(corpus)))
    best_sim = -1.0
    best_candidate = None
    for cand in candidates:
        cand_emb = average_embedding(cand)
        if cand_emb is None:
            continue
        sim = torch.dot(q_emb, cand_emb).item()  # cosine similarity (since normalized)
        if sim > best_sim:
            best_sim = sim
            best_candidate = cand
    if best_candidate is None:
        return random.choice(corpus)
    return best_candidate

def generate_triplets(train_dataset, max_samples=50000):
    triplets = []
    num_samples = min(max_samples, len(train_dataset))
    for i in range(num_samples):
        sample = train_dataset[i]
        query = sample.get("query", "")
        pos_list = extract_positives(sample)
        if not query or not pos_list:
            continue
        for pos, weight in pos_list:
            triplets.append((query, pos, None, weight))  # negative to be filled later.
    return triplets

def assign_negatives(triplets, negative_corpus, vocab_dictionary, embedding_matrix):
    new_triplets = []
    for (query, pos, _, weight) in triplets:
        neg = sample_hard_negative(query, pos, negative_corpus, vocab_dictionary, embedding_matrix)
        new_triplets.append((query, pos, neg, weight))
    return new_triplets

def load_pretrained_embeddings():
    import gensim.downloader as api
    print("Loading pretrained Word2Vec (Google News 300)...")
    w2v_model = api.load('word2vec-google-news-300')
    embedding_dim = w2v_model.vector_size
    vocab_list = w2v_model.index_to_key

    temp_dict = {}
    for word in vocab_list:
        lw = word.lower()
        if lw not in temp_dict:
            temp_dict[lw] = word

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

def average_embedding(text, vocab_dictionary, embedding_matrix):
    tokens = simple_tokenize(text)
    indices = tokens_to_indices(tokens, vocab_dictionary)
    if len(indices) == 0:
        return None
    vecs = embedding_matrix[indices]
    avg = vecs.mean(dim=0)
    norm = avg.norm() + 1e-8
    return avg / norm

def evaluate_model(model, val_dataset, vocab_dictionary, embedding_matrix, max_length, device, num_queries=200):
    import torch.nn.functional as F
    # Build candidate corpus from first num_queries validation samples.
    candidates = []
    ground_truth = []
    for i in range(min(num_queries, len(val_dataset))):
        sample = val_dataset[i]
        query = sample.get("query", "")
        pos = extract_positives(sample)
        if not query or not pos:
            continue
        # For evaluation, take the first positive as ground truth.
        candidates.append(pos[0][0])
        ground_truth.append(pos[0][0])
    if not candidates:
        return 0.0
    candidate_embs = []
    for doc in candidates:
        emb = average_embedding(doc, vocab_dictionary, embedding_matrix)
        if emb is not None:
            candidate_embs.append(emb)
    if not candidate_embs:
        return 0.0
    candidate_embs = torch.stack(candidate_embs, dim=0)  # shape: [num_candidates, emb_dim]

    mrr_total = 0.0
    count = 0
    for i in range(min(num_queries, len(val_dataset))):
        sample = val_dataset[i]
        query = sample.get("query", "")
        pos = extract_positives(sample)
        if not query or not pos:
            continue
        true_doc = pos[0][0]
        q_emb = average_embedding(query, vocab_dictionary, embedding_matrix)
        if q_emb is None:
            continue
        sims = torch.matmul(candidate_embs, q_emb.unsqueeze(1)).squeeze(1)
        sorted_indices = torch.argsort(sims, descending=True)
        # Find rank of the true document in the candidate set.
        try:
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
            mrr_total += 1.0 / rank
            count += 1
        except Exception as e:
            continue
    mrr = mrr_total / count if count > 0 else 0.0
    return mrr

##########################
# Hyperparameter Configurations
##########################
hyperparameter_configs = [
    {"lr": 0.001, "margin": 0.2, "freeze_embeddings": False, "batch_size": 64, "num_epochs": 5},
    {"lr": 0.0005, "margin": 0.2, "freeze_embeddings": False, "batch_size": 64, "num_epochs": 10},
    {"lr": 0.001, "margin": 0.3, "freeze_embeddings": False, "batch_size": 128, "num_epochs": 5},
    {"lr": 0.001, "margin": 0.2, "freeze_embeddings": True,  "batch_size": 64, "num_epochs": 5},
    {"lr": 0.002, "margin": 0.2, "freeze_embeddings": False, "batch_size": 32, "num_epochs": 5}
]

def run_experiment(config, train_dataset, val_dataset, vocab_dictionary, embedding_matrix):
    print("\n\n=== Running Configuration ===")
    print(config)
    # Build negative corpus and generate triplets from training data.
    negative_corpus = build_negative_corpus(train_dataset)
    triplets = generate_triplets(train_dataset, max_samples=50000)
    # Use hard negative sampling.
    triplets = assign_negatives(triplets, negative_corpus, vocab_dictionary, embedding_matrix)
    print("Generated", len(triplets), "triplets for training.")

    # Create Dataset and DataLoader.
    max_length = 128
    dataset_obj = MSMarcoTripletDataset(triplets, vocab_dictionary, max_length=max_length)
    dataloader = DataLoader(dataset_obj, batch_size=config["batch_size"], shuffle=True, num_workers=0)

    # Instantiate the model.
    hidden_dim = 128
    model = TwoTowerModel(vocab_size=len(vocab_dictionary),
                          embedding_dim=embedding_matrix.shape[1],
                          hidden_dim=hidden_dim,
                          pretrained_embeddings=embedding_matrix,
                          freeze_embeddings=config["freeze_embeddings"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.TripletMarginLoss(margin=config["margin"], p=2, reduction='none')

    # Training loop.
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0.0
        for batch_idx, (q, p, n, w) in enumerate(dataloader):
            q, p, n, w = q.to(device), p.to(device), n.to(device), w.to(device)
            optimizer.zero_grad()
            q_enc, pos_enc, neg_enc = model(q, p, n)
            losses = criterion(q_enc, pos_enc, neg_enc)  # shape: [batch]
            weighted_loss = (losses * w).mean()
            weighted_loss.backward()
            optimizer.step()
            total_loss += weighted_loss.item()
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {weighted_loss.item():.4f}")
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

    # Evaluate on validation set.
    mrr = evaluate_model(model, val_dataset, vocab_dictionary, embedding_matrix, max_length, device, num_queries=200)
    print(f"Validation MRR: {mrr:.4f}")

    # Save model and embeddings.
    model_path = f"ms_marco_two_tower_{config['lr']}_{config['margin']}_{config['freeze_embeddings']}_{config['batch_size']}_{config['num_epochs']}.pt"
    torch.save(model.state_dict(), model_path)
    print("Model saved as", model_path)
    return mrr

def main():
    # Step 1: Load MS MARCO dataset.
    print("Loading MS MARCO dataset...")
    train_dataset, val_dataset, test_dataset = load_ms_marco()
    print("Train dataset size:", len(train_dataset))
    
    debug_samples(train_dataset, num_samples=3)
    
    # Step 2: Load pretrained embeddings.
    global vocab_dictionary
    vocab_dictionary, embedding_matrix = load_pretrained_embeddings()

    # Run experiments for each configuration.
    results = []
    for config in hyperparameter_configs:
        mrr = run_experiment(config, train_dataset, val_dataset, vocab_dictionary, embedding_matrix)
        results.append((config, mrr))
    
    print("\n=== Experiment Results ===")
    for config, mrr in results:
        print(f"Config: {config} => Validation MRR: {mrr:.4f}")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()