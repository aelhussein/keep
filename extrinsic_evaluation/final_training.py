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
from collections import defaultdict
N_RUNS = 10

def load_best_parameters(outcome, subsample, freeze):
    """Load the best learning rates and epochs from the tuning results"""
    if subsample:
        tuning_path = f'{WORKING_DIR}/extrinsic_evaluation/results/outcome_{outcome}_lr_tuning_subsampled'
    else:
        tuning_path = f'{WORKING_DIR}/extrinsic_evaluation/results/outcome_{outcome}_lr_tuning'
    if freeze:
        tuning_path = f'{tuning_path}.pkl'
    else:
        tuning_path = f'{tuning_path}_emb_trained.pkl'
    
    with open(tuning_path, 'rb') as f:
        tuning_results = pickle.load(f)
    
    best_params = {}
    for embed_name, lr_results in tuning_results.items():
        # Find the LR with best validation loss
        best_lr, best_result = min(lr_results.items(), 
                                 key=lambda x: x[1]['best_val_loss'])
        best_params[embed_name] = {
            'learning_rate': best_lr,
            'num_epochs': best_result['total_epochs'],  # Use the number of epochs from best run
            'best_val_metrics': {
                'loss': best_result['best_val_loss'],
                'f1': max(best_result['val_f1_weighted']),
                'auc': max(best_result['val_auc']),
                'auprc': max(best_result['val_auprc']),
                'accuracy': max(best_result['val_accuracy'])
            }
        }
    
    return best_params


def train_final_model(embedding_path, learning_rate, num_epochs, device, subsample, freeze, seed, pooling_type):
    """Train final model with a specific random seed"""
    set_all_seeds(seed)
    
    # Load embeddings
    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)
    embedding_diseases = torch.tensor(embeddings, dtype=torch.float32)
    
    if any(model in embedding_path for model in ['bioclin_bert', 'clin_bert', 'biogpt']):
        embedding_diseases = F.normalize(embedding_diseases, p=2, dim=1)  # L2 norm embedding as suggested by simclr for large embeddings
    zero_vector = torch.zeros(1, embedding_diseases.size(1))
    embedding_tensor = torch.cat([embedding_diseases, zero_vector], dim=0)
    
    # Load data
    train_loader, test_loader, weights = load_data(outcome, val_split = False, subsample_majority=subsample)
    
    # Initialize model
    n_embd = embedding_diseases.size(1)
    model = TransformerModel(vocab_size, id_dict, n_embd, n_head, n_layer, num_classes, 
                            dropout=dropout, embedding_weights=embedding_tensor, freeze_emb=freeze,
                            pooling_type = pooling_type).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    
    optimizer = torch.optim.AdamW(
        [ {"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": 0.01}],
        lr=learning_rate,)
    
    # Training history
    history = {
        'train_loss': [],
        'test_metrics': None,
        'total_epochs': 0,
        'seed': seed
    }

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_sentences, batch_labels in train_loader:
            batch_sentences = batch_sentences.to(device)
            batch_labels = batch_labels.to(device)
            
            logits = model(batch_sentences)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        if (epoch + 1) % eval_interval == 0:
            print(f'Run {seed} - Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_train_loss:.4f}', flush=True)
    
    # Final evaluation on test set
    model.eval()
    test_metrics = evaluate(model, test_loader, criterion)
    history['test_metrics'] = test_metrics
    history['total_epochs'] = num_epochs
    
    return history, model

def calculate_confidence_intervals(results_list, alpha=0.05):
    # Define all metrics to analyze
    metrics = ['loss', 'f1_macro', 'f1_weighted', 'auc', 'auprc', 'accuracy']
    ci_results = {}
    
    # Set random seed for reproducibility
    np.random.seed(42)

    for metric in metrics:
        values = [run['test_metrics'][metric] for run in results_list]
        bootstrapped_means = []
        bootstrapped_medians = []  # Added median calculation

        # Perform bootstrapping
        num_bootstrap = 1000
        n_samples = len(values)
        
        for _ in range(num_bootstrap):
            bootstrap_sample = np.random.choice(values, size=n_samples, replace=True)
            bootstrapped_means.append(np.mean(bootstrap_sample))
            bootstrapped_medians.append(np.median(bootstrap_sample))
        
        # Calculate statistics
        mean = np.mean(values)  # Use original values for point estimate
        median = np.median(values)  # Added median
        lower = np.percentile(bootstrapped_means, alpha/2 * 100)
        upper = np.percentile(bootstrapped_means, (1-alpha/2) * 100)
        std = np.std(values)  # Use original values for std
        
        # Store results
        ci_results[metric] = {
            'mean': mean,
            'median': median,
            'lower': lower,
            'upper': upper,
            'std': std,
            'raw_values': values
        }
    
    return ci_results

class ResultPaths:
    """Manages paths for saving results and models"""
    def __init__(self, working_dir, outcome, subsample, freeze, overwrite):
        self.base_dir = working_dir
        self.outcome = outcome
        self.subsample = subsample
        self.freeze = freeze
        self.overwrite = overwrite
        
    def _get_base_suffix(self):
        suffix = ''
        if self.subsample:
            suffix += '_subsampled'
        if not self.freeze:
            suffix += '_emb_trained'
        return suffix
        
    def get_versioned_paths(self):
        base_path = f'{self.base_dir}/extrinsic_evaluation/results/outcome_{self.outcome}_results{self._get_base_suffix()}'
        
        if self.overwrite:
            save_path = f'{base_path}.pkl'
        else:
            # Find existing versions
            search_pattern = f'outcome_{self.outcome}_results{self._get_base_suffix()}'
            existing_files = [
                f for f in os.listdir(f'{self.base_dir}/extrinsic_evaluation/results')
                if f.startswith(search_pattern) and '_v' in f and f.endswith('.pkl')
            ]
            
            if not existing_files:
                save_path = f'{base_path}_v1.pkl'
            else:
                versions = [int(f.split('_v')[-1].replace('.pkl', '')) for f in existing_files]
                next_version = max(versions) + 1
                save_path = f'{base_path}_v{next_version}.pkl'
        
        model_save_path = save_path.replace('/results/', '/models/').replace('.pkl', '.pth')
        return save_path, model_save_path
    
    def get_paths(self):
        """Get paths for summary and detailed results based on versioned path"""
        main_save_path, model_save_path = self.get_versioned_paths()
        
        # Remove .pkl extension to get base path
        base_path = main_save_path.rsplit('.pkl', 1)[0]
        
        # Create summary and detailed paths with same versioning
        summary_path = f"{base_path}_summary.txt"
        detailed_path = f"{base_path}_detailed.pkl"
        
        return summary_path, detailed_path, model_save_path

def save_intermediate_results(results, paths, current_run, total_runs, embed_name):
    """Save intermediate results and print progress"""
    save_interval = 10  # Save every 10 runs
    summary_path, detailed_path, _ = paths.get_paths()
    if (current_run + 1) % save_interval == 0 or (current_run + 1) == total_runs:
        print(f"\nSaving intermediate results after run {current_run + 1}", flush=True)
        
        # Save results
        with open(detailed_path, 'wb') as f:
            pickle.dump(results, f)
            
        # Calculate and print intermediate statistics
        if embed_name in results:
            # Pass the individual_runs list instead of the whole results dictionary
            ci_results = calculate_confidence_intervals(results[embed_name]['individual_runs'])
            print(f"\nIntermediate results for {embed_name} after {current_run + 1} runs:")
            for metric, stats in ci_results.items():
                print(f"\n{metric.upper()}:")
                print(f"  Current Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"  Current 95% CI: [{stats['lower']:.4f}, {stats['upper']:.4f}]")

def save_final_results(results, paths, n_runs):
    """Save final summary and detailed results"""
    summary_path, detailed_path, _ = paths.get_paths()
    
    # Save summary
    with open(summary_path, 'w') as f:
        f.write(f"Bootstrap Analysis Results for Outcome {paths.outcome}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration:\n")
        f.write(f"  Number of runs: {n_runs}\n")
        f.write(f"  Subsampling: {paths.subsample}\n")
        f.write(f"  Freeze embeddings: {paths.freeze}\n\n")
        
        for embed_name, result in results.items():
            if 'error' in result:
                f.write(f"\n{embed_name}: Error - {result['error']}\n")
            else:
                f.write(f"\n{embed_name}:\n")
                for metric, stats in result['confidence_intervals'].items():
                    f.write(f"\n{metric.upper()}:\n")
                    f.write(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                    f.write(f"  Median: {stats['median']:.4f}\n")
                    f.write(f"  95% CI: [{stats['lower']:.4f}, {stats['upper']:.4f}]\n")
                f.write("-"*40 + "\n")
    
    # Save detailed results
    with open(detailed_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nFinal results saved to:\n  {summary_path}\n  {detailed_path}")

def run_final_training(outcome, subsample, freeze, load, overwrite, n_runs, pooling_type):
    """Run final training with periodic result saving"""
    paths = ResultPaths(WORKING_DIR, outcome, subsample, freeze, overwrite)
    best_params = load_best_parameters(outcome, subsample, freeze)
    summary_path, detailed_path, model_save_path = paths.get_paths()
    # Initialize empty results
    results = {}
    
    # Load and validate existing results if available
    if load and os.path.exists(detailed_path):
        with open(detailed_path, 'rb') as f:
            loaded_results = pickle.load(f)
            
        # Only keep results that have the correct structure
        for embed_name, embed_results in loaded_results.items():
            if isinstance(embed_results, dict) and \
               'individual_runs' in embed_results and \
               isinstance(embed_results['individual_runs'], list):
                results[embed_name] = embed_results
            else:
                print(f"Invalid structure for {embed_name}, will retrain from beginning", flush=True)
                results[embed_name] = {
                    'confidence_intervals': {},
                    'individual_runs': []
                }
        print(f"Loaded existing results from {detailed_path}", flush=True)
    
    for embed_name, embed_path in path_embed.items():
        print(f"\nTraining final model for {embed_name} with {n_runs} runs", flush=True)
        print("Parameters:", flush=True)
        print(f"  Learning Rate: {best_params[embed_name]['learning_rate']}", flush=True)
        print(f"  Number of Epochs: {best_params[embed_name]['num_epochs']}", flush=True)
        print(f"  Subsampling: {subsample}", flush=True)
        print(f"  Freeze Embeddings: {freeze}", flush=True)
        
        try:
            # Initialize if doesn't exist
            if embed_name not in results or 'gat' in embed_name:
                results[embed_name] = {
                    'confidence_intervals': {},
                    'individual_runs': []
                }
            
            completed_runs = len(results[embed_name]['individual_runs'])
            remaining_runs = n_runs - completed_runs
            
            if remaining_runs > 0:
                print(f"Continuing from run {completed_runs + 1}", flush=True)
                
                for run in range(completed_runs, n_runs):
                    print(f"\nStarting run {run+1}/{n_runs} for {embed_name}", flush=True)
                    
                    history, model = train_final_model(
                        embedding_path=embed_path,
                        learning_rate=best_params[embed_name]['learning_rate'],
                        num_epochs=best_params[embed_name]['num_epochs'],
                        device=device,
                        subsample=subsample,
                        freeze=freeze,
                        seed=run+1,
                        pooling_type=pooling_type
                    )
                    
                    # Add run results
                    results[embed_name]['individual_runs'].append(history)
                    
                    # Save model
                    run_model_save_path = model_save_path.replace('.pth', f'_run{run+1}.pth')
                    torch.save(model.state_dict(), run_model_save_path)
                    
                    # Update confidence intervals
                    ci_results = calculate_confidence_intervals(results[embed_name]['individual_runs'])
                    results[embed_name]['confidence_intervals'] = ci_results
                    
                    # Save intermediate results
                    save_intermediate_results(results, paths, run, n_runs, embed_name)
            else:
                print(f"All {n_runs} runs completed for {embed_name}", flush=True)
            
        except Exception as e:
            print(f"Error processing {embed_name}: {str(e)}", flush=True)
            results[embed_name] = {'error': str(e)}
    
    # Save final results
    save_final_results(results, paths, n_runs)
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

    # Freeze embeddings argument
    parser.add_argument('--freeze', action='store_true', default=True,
                      help='Freeze embeddings (default: True)')
    parser.add_argument('--no-freeze', action='store_false', dest='freeze',
                      help='Train embeddings')
    
    # Load results argument
    parser.add_argument('--load', action='store_true', default=True,
                      help='Load existing results (default: True)')
    parser.add_argument('--no-load', action='store_false', dest='load',
                      help='Train all models from scratch')
    
    # Overwrite argument
    parser.add_argument('--overwrite', action='store_true', default=False,
                      help='Overwrite existing results (default: False)')
    parser.add_argument('--no-overwrite', action='store_false', dest='overwrite',
                      help='Create new versioned results file')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    outcome = args.outcome
    subsample = args.subsample
    freeze = args.freeze
    load = args.load
    overwrite = args.overwrite

    print(f"Processing outcome: {outcome}", flush=True)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
    print(f"Number of bootstrap runs: {N_RUNS}", flush=True)
    if load:
        print("Loading exisisting results", flush = True)
    else:
        print("Not loading exisisting results", flush = True)
    if subsample:
        print("Subsampling majority", flush = True)
    else:
        print("Not subsampling majority", flush = True)
    
    try:
        start_time = time.time()
        results = run_final_training(outcome, subsample, freeze, load, overwrite, N_RUNS, pooling_type)
        print(f"\nTotal training time: {time.time() - start_time:.2f} seconds", flush=True)
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}", flush=True)
        raise