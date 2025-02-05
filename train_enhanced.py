import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import gensim.downloader as api
from datasets import load_dataset
from tqdm import tqdm
import os
from datetime import datetime
from enhanced_two_tower import EnhancedTwoTower, MarginRankingLoss, EnhancedDataset, get_hard_negatives

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

def train_epoch(model, train_loader, criterion, optimizer, device, use_hard_negatives=True):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (query_emb, doc_emb) in enumerate(train_loader):
        query_emb = query_emb.to(device)
        doc_emb = doc_emb.to(device)
        
        optimizer.zero_grad()
        
        # Get embeddings
        query_vec, doc_vec = model(query_emb, doc_emb)
        
        if use_hard_negatives:
            # Get hard negatives for each query in batch
            hard_neg_vecs = []
            for i in range(len(query_vec)):
                neg_indices = get_hard_negatives(query_vec[i], doc_vec, i)
                hard_negs = doc_vec[neg_indices]
                hard_neg_vecs.append(hard_negs)
            hard_neg_vecs = torch.cat(hard_neg_vecs, dim=0)
            
            # Compute loss with both in-batch and hard negatives
            loss = criterion(query_vec, doc_vec, hard_neg_vecs)
        else:
            # Use only in-batch negatives
            loss = criterion(query_vec, doc_vec)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 20 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches

def main():
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/enhanced_run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading Word2Vec embeddings...")
    word2vec = api.load('word2vec-google-news-300')
    
    print("Loading MS MARCO dataset...")
    dataset = load_dataset("ms_marco", "v1.1")
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    
    # Model and training parameters
    hidden_dim = 512
    batch_size = 256
    num_epochs = 15
    use_hard_negatives = True
    
    # Get actual dataset size
    dataset_size = len(train_dataset)
    print(f"Total dataset size: {dataset_size}")
    
    # Adjust training parameters based on dataset size
    total_samples = min(100000, dataset_size)
    samples_per_epoch = min(50000, dataset_size // 2)
    
    print(f"Using {total_samples} total samples with {samples_per_epoch} samples per chunk")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedTwoTower(embedding_dim=300, hidden_dim=hidden_dim).to(device)
    
    criterion = MarginRankingLoss(margin=0.2, temperature=0.1)
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
            if not queries:
                continue
            
            print(f"Got {len(queries)} query-document pairs for this chunk")
            
            train_dataset_chunk = EnhancedDataset(queries, docs, word2vec)
            train_loader = DataLoader(
                train_dataset_chunk,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                drop_last=True
            )
            
            # Train on current chunk
            chunk_loss = train_epoch(
                model, 
                train_loader, 
                criterion, 
                optimizer, 
                device,
                use_hard_negatives
            )
            
            total_loss += chunk_loss
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average Training Loss: {avg_train_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
        }, f"{output_dir}/checkpoint_epoch_{epoch+1}.pt")
        
        # Save best model
        if avg_train_loss < best_val_loss:
            best_val_loss = avg_train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, f"{output_dir}/best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
        
        scheduler.step(avg_train_loss)

if __name__ == "__main__":
    main() 