import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from datetime import datetime
import os
from tqdm import tqdm

from dataset_ms_marco import MSMarcoDataset
from enhanced_two_tower import EnhancedTwoTowerModel, InfoNCELoss
from utils import load_word2vec

def main():
    # Setup logging
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/enhanced_run_{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{output_dir}/training.log'),
            logging.StreamHandler()
        ]
    )

    # Model parameters
    embedding_dim = 300
    hidden_dim = 512
    batch_size = 128
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    word2vec = load_word2vec()
    dataset = MSMarcoDataset(word2vec)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = EnhancedTwoTowerModel(embedding_dim, hidden_dim).to(device)
    criterion = InfoNCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop
    best_loss = float('inf')
    
    logging.info("Starting training...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (query_emb, doc_emb) in enumerate(dataloader):
            query_emb = query_emb.to(device)
            doc_emb = doc_emb.to(device)
            
            optimizer.zero_grad()
            query_vec, doc_vec = model(query_emb, doc_emb)
            loss = criterion(query_vec, doc_vec)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 20 == 0:
                logging.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / batch_count
        logging.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
    
    logging.info(f"Training complete. Best model saved to: {output_dir}/best_model.pt")

if __name__ == "__main__":
    main() 