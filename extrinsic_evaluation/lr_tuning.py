ROOT_DIR = '/gpfs/commons/projects/ukbb-gursoylab'
WORKING_DIR = f'{ROOT_DIR}/aelhussein'
import sys
sys.path.append(f'{WORKING_DIR}/extrinsic_evaluation')
from transformer_model import *
from configs import * #Hyperparams
from dataloader import *
import os
import pickle
import torch
import time
import numpy as np
import argparse
import random


learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]


def train_model(embedding_path, learning_rate, device, outcome, subsample, freeze, pooling_type, **kwargs):
    """Train model with specific embedding and learning rate"""
    # Load embeddings
    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)
    embedding_diseases = torch.tensor(embeddings, dtype=torch.float32)
    
    if any(model in embedding_path for model in ['bioclin_bert', 'clin_bert', 'biogpt']):
        embedding_diseases = F.normalize(embedding_diseases, p=2, dim=1)  # L2 norm embedding as suggested by simclr for large embeddings
    zero_vector = torch.zeros(1, embedding_diseases.size(1))
    embedding_tensor = torch.cat([embedding_diseases, zero_vector], dim=0)
    
    # Load data
    dataloader, dataloader_val, weights = load_data(outcome, val_split = True, subsample_majority=subsample)
    
    # Initialize model and optimizer
    n_embd = embedding_diseases.size(1)
    model = TransformerModel(vocab_size, id_dict, n_embd, n_head, n_layer, num_classes, 
                            dropout=dropout, embedding_weights = embedding_tensor, freeze_emb = freeze,
                            pooling_type = pooling_type).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(
        [ {"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": 0.01}],
        lr=learning_rate,)

    early_stopping = EarlyStopping(patience=5, min_delta = 0)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1_macro': [],
        'val_f1_weighted': [],
        'val_auc': [],
        'val_auprc':[],
        'val_accuracy': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0,
        'early_stopped': False,
        'total_epochs': 0
    }

    for epoch in range(kwargs.get('epochs', 500)):
        # Training
        model.train()
        total_loss = 0.0
        for batch_sentences, batch_labels in dataloader:
            batch_sentences = batch_sentences.to(device)
            batch_labels = batch_labels.to(device)
            
            logits = model(batch_sentences)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(dataloader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        if epoch % kwargs.get('eval_interval', 5) == 0:
            metrics = evaluate(model, dataloader_val, criterion)
            val_loss = metrics['loss']
            history['val_loss'].append(val_loss)
            history['val_f1_macro'].append(metrics['f1_macro'])
            history['val_f1_weighted'].append(metrics['f1_weighted'])
            history['val_auc'].append(metrics['auc'])
            history['val_auprc'].append(metrics['auprc'])
            history['val_accuracy'].append(metrics['accuracy'])
            
            # Update best metrics
            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
                history['best_epoch'] = epoch
            
            # Early stopping check
            if early_stopping(val_loss, model):
                print(f"Early stopping triggered at epoch {epoch}", flush = True)
                history['early_stopped'] = True
                break
    
    history['total_epochs'] = epoch + 1
    return history


def run_tuning(outcome, subsample, freeze, load, pooling_type):
    """Run experiments for all embeddings and learning rates, skipping if results exist"""
    results = {}
    
    if subsample:
        results_path = f'{WORKING_DIR}/extrinsic_evaluation/results/outcome_{outcome}_lr_tuning_subsampled'
    else:
        results_path = f'{WORKING_DIR}/extrinsic_evaluation/results/outcome_{outcome}_lr_tuning'
    if freeze:
        results_path = f'{results_path}.pkl'
    else:
        results_path = f'{results_path}_emb_trained.pkl'
    

    if load:
        # Load existing results if they exist
        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                existing_results = pickle.load(f)
        else:
            existing_results = {}
    else:
        existing_results = {}    
    
    for embed_name, embed_path in path_embed.items():
        print(f"\nProcessing embedding {embed_name}", flush=True)
        
        # Check if we already have results for this embedding
        if embed_name in existing_results and 'gat' not in embed_name:
            print(f"Found existing results for {embed_name}", flush=True)
            results[embed_name] = existing_results[embed_name]
            best_lr, best_result = min(results[embed_name].items(),
                                     key=lambda x: x[1]['best_val_loss'])
        else:
            print(f"Training {embed_name}", flush=True)
            results[embed_name] = {}
            
            # Try all learning rates for this embedding
            for lr in learning_rates:
                print(f"\nLearning rate: {lr}", flush=True)
                
                results[embed_name][lr] = train_model(
                    embedding_path=embed_path,
                    learning_rate=lr,
                    device=device,
                    epochs=epochs,
                    eval_interval=eval_interval,
                    outcome=outcome,
                    subsample=subsample,
                    freeze = freeze,
                    pooling_type = pooling_type
                )
                print(f"Validation Loss: {min(results[embed_name][lr]['val_loss']):.4f}", flush = True)
            
            # Find best results
            best_lr, best_result = min(results[embed_name].items(),
                                     key=lambda x: x[1]['best_val_loss'])
            
            # Save updated results after each embedding
            with open(results_path, 'wb') as f:
                pickle.dump({**existing_results, **results}, f)
        
        # Print best results for this embedding
        print("\n" + "="*50, flush=True)
        print(f"Results for {embed_name}:", flush=True)
        print(f"Best Learning Rate: {best_lr}", flush=True)
        print(f"Best Validation Loss: {best_result['best_val_loss']:.4f}", flush=True)
        print(f"Epochs Trained: {best_result['total_epochs']}", flush=True)
        print(f"Early Stopped: {best_result['early_stopped']}", flush=True)
        print(f"Best Validation Metrics:", flush=True)
        print(f"  F1 Score: {max(best_result['val_f1_weighted']):.4f}", flush=True)
        print(f"  AUC: {max(best_result['val_auc']):.4f}", flush=True)
        print(f"  AUPRC: {max(best_result['val_auprc']):.4f}", flush=True)
        print(f"  Accuracy: {max(best_result['val_accuracy']):.4f}", flush=True)
        print("="*50 + "\n", flush=True)
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='Train models for different outcomes')
    parser.add_argument('--outcome', type=int, required=True,
                      help='Outcome code to process')
    # Subsample argument
    parser.add_argument('--subsample', action='store_true', default=False,
                      help='Subsample majority (default: True)')
    parser.add_argument('--no-subsample', action='store_false', dest='subsample',
                      help='Do not subsample majority')
    #Freeze embeddings
    parser.add_argument('--freeze', action='store_true', default=True,
                      help='Freeze embeddings (default: True)')
    parser.add_argument('--no-freeze', action='store_false', dest='freeze',
                      help='Train embeddings')
    #load existing results
    parser.add_argument('--load', action='store_true', default=True,
                      help='Load existing results (default: True)')
    parser.add_argument('--no-load', action='store_false', dest='load',
                      help='Train all models from scratch')
    return parser.parse_args()

if __name__ == "__main__":
    set_all_seeds()
    args = parse_args()
    outcome = args.outcome
    subsample = args.subsample
    freeze = args.freeze
    load = args.load
    
    print(f"Processing outcome: {outcome}", flush=True)
    if freeze:
        print(f"Embeddings frozen", flush=True)
    else:
        print(f"Embeddings trained", flush=True)
    if load:
        print(f"Existing results loaded", flush=True)
    else:
        print(f"Existing results not loaded", flush=True)

    start_time = time.time()
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
    
    try:
        results = run_tuning(outcome, subsample,freeze, load, pooling_type)
        print(f"\nTotal time: {time.time() - start_time:.2f} seconds", flush=True)
    except Exception as e:
        print(f"Error in main execution: {str(e)}", flush=True)
        raise