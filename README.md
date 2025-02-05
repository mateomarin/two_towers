# Two-Tower Model for Document Retrieval

A neural information retrieval system using a two-tower architecture to match queries with relevant documents.

## Core Files
- `enhanced_two_tower.py` - Enhanced model with hard negative mining and margin loss
- `simple_two_tower.py` - Original implementation for comparison
- `train_enhanced.py` - Training script with improved data handling
- `validate_enhanced.py` - Validation script with MRR metrics
- `compare_models.py` - Utilities to compare model performances

## Usage

1. Train the enhanced model:
bash
python train_enhanced.py

2. Validate model performance:
bash
python validate_enhanced.py --model_path output/enhanced_run_TIMESTAMP/best_model.pt --output_dir results

3. Compare models:
bash
python compare_models.py

## Model Architecture
- Bidirectional GRU encoders
- Enhanced projection networks with LayerNorm
- Hard negative mining
- Margin-based contrastive loss
- In-batch negatives

## Requirements
- PyTorch
- gensim (for Word2Vec embeddings)
- datasets (Hugging Face)
- tqdm
- tabulate