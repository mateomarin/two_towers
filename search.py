import os
import torch
import json
import numpy as np
import torch.nn.functional as F
from two_tower_model import TwoTowerModel
from tokenizer import simple_tokenize, tokens_to_indices

# ----- Configuration: set the folder containing your experiment outputs -----
# Change this to the folder where your training script saved the outputs.
OUTPUT_DIR = "output/lr_0.001_margin_0.2_freeze_False_bs_64_ep_5"

MODEL_PATH = os.path.join(OUTPUT_DIR, "model.pt")
VOCAB_PATH = os.path.join(OUTPUT_DIR, "vocab.json")  # or "ms_marco_word_to_index.json" if you saved it that way
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "embeddings.npy")
# For candidate documents, you can either supply a file or use a hard-coded list.
# Here we assume a file named "candidate_docs.txt" exists in the same output folder.
CANDIDATE_DOCS_PATH = os.path.join(OUTPUT_DIR, "candidate_docs.txt")

# ----- Model and Inference Settings -----
MAX_LENGTH = 128
TOP_K = 5
HIDDEN_DIM = 128  # Must match the value used during training

# ----- Load Saved Resources -----
# Load vocabulary.
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab_dictionary = json.load(f)

# Load embeddings.
embedding_matrix = np.load(EMBEDDINGS_PATH)
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)

# Determine vocabulary size and embedding dimension.
vocab_size = len(vocab_dictionary)
embedding_dim = embedding_matrix.shape[1]

# Instantiate the Two Tower Model.
# Use the same settings as during training (if you fine-tuned, set freeze_embeddings accordingly).
model = TwoTowerModel(vocab_size=vocab_size, embedding_dim=embedding_dim,
                      hidden_dim=HIDDEN_DIM, pretrained_embeddings=embedding_matrix,
                      freeze_embeddings=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ----- Helper Functions for Inference -----
def encode_query(query):
    """Tokenize, pad/truncate, and encode the query using the query encoder."""
    tokens = simple_tokenize(query)
    indices = tokens_to_indices(tokens, vocab_dictionary)
    if len(indices) < MAX_LENGTH:
        indices += [0] * (MAX_LENGTH - len(indices))
    else:
        indices = indices[:MAX_LENGTH]
    query_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        query_embedding = model.encode_query(query_tensor)
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    return query_embedding

def load_candidate_documents(path):
    """Load candidate documents from a text file (one document per line)."""
    if not os.path.exists(path):
        print(f"Candidate documents file not found at {path}.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
    return docs

def compute_candidate_embeddings(docs):
    """Compute embeddings for a list of candidate documents using the document encoder."""
    embeddings = []
    for doc in docs:
        tokens = simple_tokenize(doc)
        indices = tokens_to_indices(tokens, vocab_dictionary)
        if len(indices) < MAX_LENGTH:
            indices += [0] * (MAX_LENGTH - len(indices))
        else:
            indices = indices[:MAX_LENGTH]
        doc_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            doc_embedding = model.encode_doc(doc_tensor)
        doc_embedding = F.normalize(doc_embedding, p=2, dim=1)
        embeddings.append(doc_embedding.squeeze(0))
    if embeddings:
        return torch.stack(embeddings)
    else:
        return None

def similarity_search(query, candidate_embs, candidate_docs, top_k=TOP_K):
    """Given a query, compute its embedding and return the top-k similar candidate documents."""
    query_emb = encode_query(query)
    sims = torch.matmul(query_emb, candidate_embs.t()).squeeze(0)
    topk = torch.topk(sims, k=top_k)
    results = []
    for idx, score in zip(topk.indices, topk.values):
        results.append((candidate_docs[idx.item()], score.item()))
    return results

# ----- Main Inference Loop -----
if __name__ == "__main__":
    # Load candidate documents.
    candidate_docs = load_candidate_documents(CANDIDATE_DOCS_PATH)
    if not candidate_docs:
        # If no candidate document file exists, you can hard-code a few examples or use a subset of training positives.
        print("No candidate documents found. Please provide a candidate_docs.txt file in the output folder.")
        candidate_docs = [
            "Example candidate document text 1.",
            "Example candidate document text 2.",
            "Example candidate document text 3."
        ]
    print(f"Loaded {len(candidate_docs)} candidate documents.")

    # Compute candidate embeddings.
    candidate_embs = compute_candidate_embeddings(candidate_docs)
    if candidate_embs is None:
        print("Failed to compute candidate embeddings.")
        exit(1)
    print("Candidate document embeddings computed.")

    # Interactive query loop.
    print("\n=== Similarity Search ===")
    while True:
        query = input("Enter a query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        results = similarity_search(query, candidate_embs, candidate_docs, top_k=TOP_K)
        print("\nTop matching documents:")
        for doc, score in results:
            print(f"Score: {score:.4f} | {doc[:200]}...")
        print("\n")