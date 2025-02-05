import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gensim.downloader as api
from datasets import load_dataset
import numpy as np
from typing import List, Tuple
import random
from tqdm import tqdm
import os
from datetime import datetime

class SimpleTwoTower(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.query_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,  # Increased depth
            batch_first=True,
            bidirectional=True,
            dropout=0.1  # Added dropout
        )
        self.doc_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # Increased projection network capacity
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.doc_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
    def encode_query(self, query_emb):
        _, query_hidden = self.query_encoder(query_emb)
        query_hidden = torch.cat((query_hidden[-2], query_hidden[-1]), dim=1)
        query_vec = self.query_proj(query_hidden)
        return nn.functional.normalize(query_vec, p=2, dim=1)
    
    def encode_doc(self, doc_emb):
        _, doc_hidden = self.doc_encoder(doc_emb)
        doc_hidden = torch.cat((doc_hidden[-2], doc_hidden[-1]), dim=1)
        doc_vec = self.doc_proj(doc_hidden)
        return nn.functional.normalize(doc_vec, p=2, dim=1)
        
    def forward(self, query_emb, doc_emb):
        query_vec = self.encode_query(query_emb)
        doc_vec = self.encode_doc(doc_emb)
        return query_vec, doc_vec

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, query_vec, doc_vec):
        sim_matrix = torch.matmul(query_vec, doc_vec.t()) / self.temperature
        labels = torch.arange(len(query_vec), device=query_vec.device)
        loss_q2d = nn.functional.cross_entropy(sim_matrix, labels)
        loss_d2q = nn.functional.cross_entropy(sim_matrix.t(), labels)
        return (loss_q2d + loss_d2q) / 2

class SimpleDataset(Dataset):
    def __init__(self, queries: List[str], docs: List[str], word2vec, max_length: int = 30):
        self.queries = queries
        self.docs = docs
        self.word2vec = word2vec
        self.max_length = max_length
        self.embedding_dim = word2vec.vector_size
        
    def text_to_embedding(self, text: str) -> torch.Tensor:
        words = text.lower().split()[:self.max_length]
        embeddings = []
        
        for word in words:
            try:
                emb = self.word2vec[word]
                embeddings.append(emb)
            except KeyError:
                continue
        
        if not embeddings:
            embeddings = [np.zeros(self.embedding_dim)]
            
        embeddings = np.array(embeddings)
        
        if len(embeddings) < self.max_length:
            padding = np.zeros((self.max_length - len(embeddings), self.embedding_dim))
            embeddings = np.vstack([embeddings, padding])
        else:
            embeddings = embeddings[:self.max_length]
            
        return torch.tensor(embeddings, dtype=torch.float32)
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query_emb = self.text_to_embedding(self.queries[idx])
        doc_emb = self.text_to_embedding(self.docs[idx])
        return query_emb, doc_emb

def prepare_data(dataset, num_samples=50000, offset=0):
    queries = []
    docs = []
    
    print(f"Preparing {num_samples} pairs starting from offset {offset}...")
    for sample in tqdm(dataset.select(range(offset, offset + num_samples))):
        query = sample.get("query", "")
        if not query or "passages" not in sample:
            continue
            
        passages = sample["passages"]
        passage_texts = passages.get("passage_text", [])
        is_selected = passages.get("is_selected", [])
        
        for text, selected in zip(passage_texts, is_selected):
            if selected == 1:
                queries.append(query)
                docs.append(text)
    
    return queries, docs

def evaluate_model(model, val_dataset, word2vec, device, num_samples=1000):
    model.eval()
    val_queries, val_docs = prepare_data(val_dataset, num_samples=num_samples)
    dataset = SimpleDataset(val_queries, val_docs, word2vec)
    val_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    total_loss = 0
    criterion = InfoNCELoss()
    
    with torch.no_grad():
        for query_emb, doc_emb in val_loader:
            query_emb = query_emb.to(device)
            doc_emb = doc_emb.to(device)
            query_vec, doc_vec = model(query_emb, doc_emb)
            loss = criterion(query_vec, doc_vec)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main():
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading Word2Vec embeddings...")
    word2vec = api.load('word2vec-google-news-300')
    
    print("Loading MS MARCO dataset...")
    dataset = load_dataset("ms_marco", "v1.1")
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    
    # Model and training parameters
    hidden_dim = 512  # Increased hidden dimension
    batch_size = 256
    num_epochs = 15
    
    # Get actual dataset size
    dataset_size = len(train_dataset)
    print(f"Total dataset size: {dataset_size}")
    
    # Adjust training parameters based on dataset size
    total_samples = min(100000, dataset_size)  # Cap at dataset size
    samples_per_epoch = min(50000, dataset_size // 2)  # Use half dataset per chunk
    
    print(f"Using {total_samples} total samples with {samples_per_epoch} samples per chunk")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleTwoTower(embedding_dim=300, hidden_dim=hidden_dim).to(device)
    
    criterion = InfoNCELoss(temperature=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    print("Starting training...")
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Process data in chunks
        for chunk_start in range(0, total_samples, samples_per_epoch):
            chunk_size = min(samples_per_epoch, total_samples - chunk_start)
            if chunk_size <= 0:
                break
                
            print(f"\nProcessing chunk from {chunk_start} to {chunk_start + chunk_size}")
            queries, docs = prepare_data(train_dataset, num_samples=chunk_size, offset=chunk_start)
            if not queries:  # Skip if no data
                continue
            
            print(f"Got {len(queries)} query-document pairs for this chunk")
            
            train_dataset_chunk = SimpleDataset(queries, docs, word2vec)
            train_loader = DataLoader(
                train_dataset_chunk,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                drop_last=True
            )
            
            # Training loop for current chunk
            for batch_idx, (query_emb, doc_emb) in enumerate(train_loader):
                query_emb = query_emb.to(device)
                doc_emb = doc_emb.to(device)
                
                optimizer.zero_grad()
                query_vec, doc_vec = model(query_emb, doc_emb)
                
                loss = criterion(query_vec, doc_vec)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 20 == 0:
                    print(f"Epoch {epoch+1}, Chunk {chunk_start//samples_per_epoch + 1}, "
                          f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        if num_batches > 0:  # Only compute average if we processed some batches
            avg_train_loss = total_loss / num_batches
            print(f"Epoch {epoch+1} completed. Average Training Loss: {avg_train_loss:.4f}")
            
            # Validation
            val_loss = evaluate_model(model, val_dataset, word2vec, device)
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model and early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, f"{output_dir}/best_model.pt")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
    
    # After loading best model and before validation index preparation
    checkpoint = torch.load(f"{output_dir}/best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    def encode_text(text: str, encoder_type='query') -> torch.Tensor:
        dataset = SimpleDataset([text], [text], word2vec)
        emb = dataset[0][0].unsqueeze(0).to(device)
        with torch.no_grad():
            if encoder_type == 'query':
                vec = model.encode_query(emb)
            else:
                vec = model.encode_doc(emb)
        return vec.cpu()
    
    print("\nPreparing MS MARCO validation index...")
    # Get a subset of MS MARCO validation documents for testing
    val_docs = []
    val_doc_to_queries = {}  # Map documents to their relevant queries
    query_to_relevant = {}   # Map queries to their relevant documents
    
    # Process validation set to build index
    num_val_docs = 10000  # Number of validation documents to index
    for sample in tqdm(val_dataset.select(range(num_val_docs))):
        query = sample.get("query", "")
        if not query or "passages" not in sample:
            continue
            
        passages = sample["passages"]
        passage_texts = passages.get("passage_text", [])
        is_selected = passages.get("is_selected", [])
        
        query_to_relevant[query] = []
        for text, selected in zip(passage_texts, is_selected):
            if text not in val_docs:
                val_docs.append(text)
            if selected == 1:
                query_to_relevant[query].append(text)
                if text not in val_doc_to_queries:
                    val_doc_to_queries[text] = []
                val_doc_to_queries[text].append(query)
    
    print(f"Built index with {len(val_docs)} documents")
    
    # Encode all validation documents
    print("Encoding validation documents...")
    doc_encodings = []
    batch_size = 128
    for i in tqdm(range(0, len(val_docs), batch_size)):
        batch_docs = val_docs[i:i + batch_size]
        batch_encodings = []
        for doc in batch_docs:
            doc_vec = encode_text(doc, 'doc')
            batch_encodings.append(doc_vec)
        doc_encodings.extend(batch_encodings)
    doc_encodings = torch.cat(doc_encodings, dim=0)
    
    def search_ms_marco(query: str, k: int = 5):
        query_vec = encode_text(query, 'query')
        similarities = torch.nn.functional.cosine_similarity(query_vec, doc_encodings)
        top_k = torch.topk(similarities, k=k)
        results = []
        for idx, score in zip(top_k.indices, top_k.values):
            results.append((val_docs[idx], score.item()))
        return results
    
    # Test with some MS MARCO validation queries
    print("\nTesting with MS MARCO validation queries...")
    num_test_queries = 5
    test_count = 0
    mrr_sum = 0
    
    with open(f"{output_dir}/ms_marco_results.txt", "w") as f:
        f.write("MS MARCO Validation Results\n")
        f.write("=========================\n\n")
        
        for sample in val_dataset:
            query = sample.get("query", "")
            if not query in query_to_relevant or not query_to_relevant[query]:
                continue
                
            relevant_docs = set(query_to_relevant[query])
            results = search_ms_marco(query, k=10)
            
            f.write(f"\nQuery: {query}\n")
            f.write("Relevant documents: {}\n".format(len(relevant_docs)))
            
            # Calculate MRR
            mrr = 0
            for rank, (doc, score) in enumerate(results, 1):
                is_relevant = "✓" if doc in relevant_docs else " "
                f.write(f"{rank}. [{is_relevant}] ({score:.4f}) {doc[:200]}...\n")
                
                if doc in relevant_docs and mrr == 0:
                    mrr = 1.0 / rank
            
            mrr_sum += mrr
            test_count += 1
            
            print(f"\nQuery: {query}")
            print(f"MRR: {mrr:.4f}")
            for rank, (doc, score) in enumerate(results[:3], 1):
                is_relevant = "✓" if doc in relevant_docs else " "
                print(f"{rank}. [{is_relevant}] ({score:.4f}) {doc[:100]}...")
            
            if test_count >= num_test_queries:
                break
        
        avg_mrr = mrr_sum / test_count
        print(f"\nAverage MRR over {test_count} queries: {avg_mrr:.4f}")
        f.write(f"\nAverage MRR over {test_count} queries: {avg_mrr:.4f}\n")
    
    # Continue with the curated test corpus...
    
    # Save test corpus
    test_corpus = [
        "Coffee is a brewed drink prepared from roasted coffee beans.",
        "Paris is the capital and largest city of France.",
        "Python is a popular programming language.",
        "Common COVID-19 symptoms include fever and cough.",
        "A chocolate cake is a cake flavored with melted chocolate.",
        "The Earth is the third planet from the Sun.",
        "Machine learning is a subset of artificial intelligence.",
        "The Great Wall of China is over 13,000 miles long.",
        "Photosynthesis is how plants convert sunlight into energy.",
        "The human body has 206 bones."
    ]
    
    with open(f"{output_dir}/test_corpus.txt", "w") as f:
        for doc in test_corpus:
            f.write(doc + "\n")
    
    # Test queries
    test_queries = [
        "how to make coffee",
        "what is the capital of France",
        "best programming languages",
        "covid symptoms",
        "chocolate cake recipe"
    ]
    
    # Save test results
    with open(f"{output_dir}/test_results.txt", "w") as f:
        f.write("Test Results\n")
        f.write("============\n\n")
        
        for query in test_queries:
            f.write(f"\nQuery: {query}\n")
            query_vec = encode_text(query, 'query')
            
            similarities = []
            for doc in test_corpus:
                doc_vec = encode_text(doc, 'doc')
                sim = torch.nn.functional.cosine_similarity(query_vec, doc_vec)
                similarities.append((doc, sim.item()))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            f.write("Top 3 matches:\n")
            for doc, sim in similarities[:3]:
                f.write(f"Similarity: {sim:.4f} | Doc: {doc}\n")
            
            # Also print to console
            print(f"\nQuery: {query}")
            print("Top 3 matches:")
            for doc, sim in similarities[:3]:
                print(f"Similarity: {sim:.4f} | Doc: {doc}")

if __name__ == "__main__":
    main() 