import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import gensim.downloader as api
import os
from tqdm import tqdm
from datetime import datetime
from margin_two_tower import TwoTowerModel, InfoNCELoss, SimpleDataset
from dataset_ms_marco import load_ms_marco_train
import argparse
from utils import load_or_cache_word2vec

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512)
    args = parser.parse_args()
    
    # Load cached Word2Vec embeddings
    word2vec = load_or_cache_word2vec()
    
    # Load MS MARCO dataset
    print("Loading MS MARCO dataset...")
    queries, docs = load_ms_marco_train()  # This function should also use caching
    
    # Model parameters
    embedding_dim = 300
    hidden_dim = args.hidden_dim
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = 0.001
    
    # Get actual dataset size
    dataset_size = len(queries)  # Use loaded queries directly
    chunk_size = 50000
    
    # Initialize model and loss
    model = TwoTowerModel(embedding_dim, hidden_dim)
    criterion = InfoNCELoss(temperature=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}/margin_run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving model checkpoints to: {output_dir}")
    
    print("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Process data in chunks up to dataset_size
        for chunk_start in range(0, dataset_size, chunk_size):
            # Adjust chunk size for last chunk
            current_chunk_size = min(chunk_size, dataset_size - chunk_start)
            
            print(f"\nProcessing chunk from {chunk_start} to {chunk_start + current_chunk_size}")
            queries_chunk = queries[chunk_start:chunk_start + current_chunk_size]
            docs_chunk = docs[chunk_start:chunk_start + current_chunk_size]
            if not queries_chunk:
                continue
            
            print(f"Got {len(queries_chunk)} query-document pairs for this chunk")
            
            train_dataset_chunk = SimpleDataset(queries_chunk, docs_chunk, word2vec)
            train_loader = DataLoader(
                train_dataset_chunk,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4
            )
            
            for batch_idx, (query_emb, doc_emb) in enumerate(train_loader):
                query_emb = query_emb.to(device)
                doc_emb = doc_emb.to(device)
                
                optimizer.zero_grad()
                query_vec, doc_vec = model(query_emb, doc_emb)
                loss = criterion(query_vec, doc_vec)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 20 == 0:
                    print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Save if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"{output_dir}/best_model.pt")

    print(f"\nTraining complete. Best model saved to: {output_dir}/best_model.pt")

if __name__ == "__main__":
    main() 