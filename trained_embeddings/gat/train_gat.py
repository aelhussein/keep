#!/gpfs/commons/home/aelhussein/anaconda3/envs/cuda_env_ne1/bin/python
#SBATCH --job-name=gat_training
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aelhussein@nygenome.org
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/outputs/output_gat_model.txt
#SBATCH --error=logs/errors/errors_gat_model.txt

## Imports
import os
os.chdir("/gpfs/commons/projects/ukbb-gursoylab/aelhussein")
import sys
sys.path.append("/gpfs/commons/projects/ukbb-gursoylab/aelhussein")
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset, RandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, flush = True)

## Choose model and specify iteration
"""
1: ROLLUP UKBB OMOP codes level 4 (UKBB)
2: ROLLUP UKBB OMOP codes level 5 (UKBB)
3: ROLLUP UKBB OMOP codes level 5 with minimum count filter (UKBB)
4: ROLLUP UKBB OMOP codes level 5 with minimum count filter and hierarchy counts (UKBB)
"""

MODEL_NUM = 1
MODEL_NUM_N2V = 1

ROLLUP_LVL = 5
DATASET = 3

TRAIN_MODE = True
INIT_EMBEDDING = True

## Hyperparameters
EMBEDDING_SIZE = 100
LR = 1e-3
NUM_EPOCHS = 100
DROPOUT = 0.3
BATCH_SIZE = 512

EMBEDDING_PATH = "trained_embeddings/gat"
EMBEDDING_PATH_N2V = "trained_embeddings/our_embeddings"
COOC_PATH = "datasets/cooc_matrices"
VOCAB_PATH = "datasets/vocab_dict"
GRAPH_PATH = '/gpfs/commons/projects/ukbb-gursoylab/aelhussein/trained_embeddings/gat/graph'

# Params
model_name_dict = {1: 'ukbb_omop_rollup_lvl_4',
                   2: 'ukbb_omop_rollup_lvl_5',
                   3: 'ukbb_omop_rollup_lvl_5_ct_filter',
                   4: 'ukbb_omop_hierarchy_rollup_lvl_5_ct_filter'}

model_name = f"gat_model_{MODEL_NUM}_emb_{EMBEDDING_SIZE}d_{model_name_dict[DATASET]}"

model_path = f'{EMBEDDING_PATH}/models/'+model_name+'.pth'


cooc_matrix_path = {1: f"{COOC_PATH}/cooc_ukbb_omop_rollup_lvl_4.pickle",
                    2: f"{COOC_PATH}/cooc_ukbb_omop_rollup_lvl_5.pickle",
                    3: f"{COOC_PATH}/cooc_ukbb_omop_rollup_lvl_5_ct_filter.pickle",
                    4: f"{COOC_PATH}/cooc_ukbb_omop_hierarchy_rollup_lvl_5_ct_filter.pickle"}

id_dict_path = {1: f'{VOCAB_PATH}/code2id_ukbb_omop_rollup_lvl_4.pickle',
                2: f'{VOCAB_PATH}/code2id_ukbb_omop_rollup_lvl_5.pickle',
                3: f'{VOCAB_PATH}/code2id_ukbb_omop_rollup_lvl_5_ct_filter.pickle',
                4: f'{VOCAB_PATH}/code2id_ukbb_omop_rollup_lvl_5_ct_filter.pickle'}

name_dict_path = {1: f'{VOCAB_PATH}/code2name_ukbb_omop_rollup_lvl_4.pickle', 
                  2: f'{VOCAB_PATH}/code2name_ukbb_omop_rollup_lvl_5.pickle',
                  3: f'{VOCAB_PATH}/code2name_ukbb_omop_rollup_lvl_5_ct_filter.pickle',
                  4: f'{VOCAB_PATH}/code2name_ukbb_omop_rollup_lvl_5_ct_filter.pickle', }

graph_matrix_path = {1: f"{GRAPH_PATH}/cooc_ukbb_omop_rollup_lvl_4.pth",
                    2: f"{GRAPH_PATH}/cooc_ukbb_omop_rollup_lvl_5.pth",
                    3: f"{GRAPH_PATH}/cooc_ukbb_omop_rollup_lvl_5_ct_filter.pth",
                    4: f"{GRAPH_PATH}/cooc_ukbb_omop_hierarchy_rollup_lvl_5_ct_filter.pth"}
graph_save_path = graph_matrix_path[DATASET]

init_node2vec_path = {1: f'{EMBEDDING_PATH_N2V}/node2vec_embeddings/n2v_model_{MODEL_NUM_N2V}_emb_{EMBEDDING_SIZE}d_ukbb_omop_rollup_lvl_4.pickle',
                      2: f'{EMBEDDING_PATH_N2V}/node2vec_embeddings/n2v_model_{MODEL_NUM_N2V}_emb_{EMBEDDING_SIZE}d_ukbb_omop_rollup_lvl_5.pickle',
                      3: f'{EMBEDDING_PATH_N2V}/node2vec_embeddings/n2v_model_{MODEL_NUM_N2V}_emb_{EMBEDDING_SIZE}d_ukbb_omop_rollup_lvl_5_ct_filter.pickle',
                      4: f'{EMBEDDING_PATH_N2V}/node2vec_embeddings/n2v_model_{MODEL_NUM_N2V}_emb_{EMBEDDING_SIZE}d_ukbb_omop_rollup_lvl_5_ct_filter.pickle'}

export_name = {1: 'ukbb_omop_rollup_lvl_4',
               2: 'ukbb_omop_rollup_lvl_5',
               3: 'ukbb_omop_rollup_lvl_5_ct_filter',
               4: 'ukbb_omop_hierarchy_rollup_lvl_5_ct_filter'}

vocabulary = {1: 'omop',
              2: 'omop',
              3: 'omop',
              4: 'omop'}


def get_edges_optimized(cooc_matrix, id_dict, threshold=10):
    """Vectorized edge creation from co-occurrence matrix"""
    diagnoses = list(id_dict.keys())
    
    # Convert to numpy arrays for fast operations
    matrix = cooc_matrix.values if isinstance(cooc_matrix, pd.DataFrame) else cooc_matrix
    diag_arr = np.array(diagnoses)
    
    # Create mask for valid edges (excluding diagonal and below threshold)
    mask = (matrix > threshold) & ~np.eye(matrix.shape[0], dtype=bool)
    
    # Get indices of valid edges using vectorized operations
    rows, cols = np.where(mask)
    
    # Get upper triangle indices only (since co-occurrence is symmetric)
    upper_mask = rows < cols
    rows = rows[upper_mask]
    cols = cols[upper_mask]
    
    # Vectorized extraction of values and diagnosis pairs
    values = matrix[rows, cols]
    diag1 = diag_arr[rows]
    diag2 = diag_arr[cols]
    
    # Stack results into edge list (using numpy for speed)
    edges = np.column_stack((diag1, diag2, values)).tolist()
    
    return [tuple(edge) for edge in edges]

def create_dataset(cooc_matrix, id_dict, embedding_size, embeddings_init=None, save_path=graph_save_path,):
    # Check if the graph data already exists
    # if os.path.exists(save_path):
    #     print(f"Loading existing graph data from {save_path}")
    #     graph_data = torch.load(save_path, weights_only=False, map_location=device)
    #     return graph_data

    print("Graph data not found. Creating new dataset...")

    
    edges = get_edges_optimized(cooc_matrix, id_dict, threshold=15)

    # Create graph using NetworkX
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    # Convert edge list to tensor format
    edge_index = torch.tensor([[id_dict[diag1], id_dict[diag2]] for diag1, diag2, _ in edges], dtype=torch.long).t()
    edge_weights = torch.tensor([w for _, _, w in edges], dtype=torch.float)
    
    edge_weights = torch.log(edge_weights + 1)  # Log scaling to preserve variation
    edge_weights = edge_weights / edge_weights.max() 


    # Initialize node features (Random or Pre-trained Node2Vec)
    diagnoses = list(id_dict.keys())
    num_diagnoses = len(diagnoses)
    if embeddings_init is not None:
        node_features = torch.tensor(embeddings_init, dtype=torch.float)
    else:
        node_features = torch.randn((num_diagnoses, embedding_size), dtype=torch.float)

    # Create PyG Data object and move to device
    #node_features = F.normalize(node_features, p=2, dim=1)
    graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weights).to(device)

    # Save the graph data for future use
    torch.save(graph_data, save_path)
    print(f"Graph data saved at {save_path}")

    return graph_data



class GAT(nn.Module):
    def __init__(self, data, in_features, hidden_features, out_features, dropout):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.data = data
        
        # Simpler design with consistent dimensions
        self.gat1 = GATConv(
            in_features, 
            hidden_features, 
            heads=1,  # Single head for simplicity
            dropout=dropout,
            add_self_loops=True,  # Critical for self-feature propagation
            edge_dim=1
        )
        
        self.gat2 = GATConv(
            hidden_features,
            out_features, 
            heads=1,
            dropout=dropout,
            add_self_loops=True,  # Critical for self-feature propagation
            edge_dim=1
        )

        self.ln1 = nn.LayerNorm(hidden_features)
        self.ln2 = nn.LayerNorm(out_features)
        self.residual = nn.Linear(in_features, out_features)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.init_weights()
    
    def init_weights(self):
        # Initialize with small weights for numerical stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self):
        x, edge_index, edge_weight = self.data.x, self.data.edge_index, self.data.edge_attr
        x_res = self.residual(self.data.x)
        # First layer with activation
        x = F.dropout(x_res, self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index, edge_weight))
        
        # Second layer
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, edge_index, edge_weight)
        x = x + (self.alpha * x_res)
        #x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def get_embeddings(self):
        return self.forward().detach().cpu().numpy()


def cooccurrence_loss(embeddings, edge_index, edge_weight, neg_samples=5):
    src_nodes = edge_index[0]
    tgt_nodes = edge_index[1]
    num_nodes = embeddings.shape[0]

    # Compute cosine similarity for positive pairs
    pos_score = (embeddings[src_nodes] * embeddings[tgt_nodes]).sum(dim=1).sigmoid()

    # Efficient negative sampling
    neg_scores = []
    for _ in range(neg_samples):
        # Sample random negatives in vectorized way
        neg_tgt = torch.randint(0, num_nodes, (len(src_nodes),), device=embeddings.device)

        # Avoid true edges by re-sampling in a single efficient step
        is_positive = (neg_tgt == tgt_nodes)  # Mask where negatives are accidentally positives
        neg_tgt[is_positive] = torch.randint(0, num_nodes, (is_positive.sum(),), device=embeddings.device)

        # Compute cosine similarity for negative pairs
        neg_score = (embeddings[src_nodes] * embeddings[neg_tgt]).sum(dim=1).sigmoid()
        neg_scores.append(neg_score)

    # Stack and compute loss
    neg_scores = torch.stack(neg_scores, dim=1)
    
    pos_loss = torch.log(pos_score + 1e-9)  # Avoid log(0)
    neg_loss = torch.log(1 - neg_scores + 1e-9).sum(dim=1)  # Sum over negative samples
    weighted_loss = (-edge_weight * (pos_loss + neg_loss)).mean()

    return weighted_loss

def variance_regularization(embeddings, min_std=0.2):
    """Much stronger variance regularization to prevent collapse"""
    # Calculate variance along each dimension
    var_per_dim = embeddings.var(dim=0)
    
    # Penalize dimensions with low variance (below threshold)
    low_var_penalty = F.relu(min_std - var_per_dim).mean()
    
    # Also encourage diversity between dimensions
    cov_matrix = torch.mm(embeddings.T, embeddings) / embeddings.size(0)
    # Remove diagonal (self-correlation)
    mask = 1 - torch.eye(cov_matrix.size(0), device=cov_matrix.device)
    off_diag = cov_matrix * mask
    # Penalize high correlation between dimensions
    correlation_penalty = (off_diag ** 2).mean()
    
    return low_var_penalty + 0.1 * correlation_penalty



def link_prediction_loss(embeddings, edge_index, edge_weight):
    pos_src, pos_tgt = edge_index

    # Positive samples (true edges)
    pos_scores = (embeddings[pos_src] * embeddings[pos_tgt]).sum(dim=1)

    # Negative sampling: Pick nodes that are NOT connected
    neg_tgt = torch.randint(0, embeddings.size(0), (pos_src.size(0),), device=embeddings.device)
    
    # Ensure negatives are not in positive edges
    while torch.any((pos_src == neg_tgt)):
        neg_tgt = torch.randint(0, embeddings.size(0), (pos_src.size(0),), device=embeddings.device)

    neg_scores = (embeddings[pos_src] * embeddings[neg_tgt]).sum(dim=1)

    # Margin Ranking Loss: Ensure positive pairs have higher similarity than negative pairs
    loss = F.margin_ranking_loss(
        pos_scores, neg_scores, 
        target=torch.ones_like(pos_scores), margin=1.0
    )

    # Weight loss based on edge strength (importance of co-occurrence)
    weighted_loss = (edge_weight * loss).mean()
    
    return weighted_loss + variance_regularization(embeddings)


def loss_function(embeddings, edge_index, edge_weight, neg_samples=10):
    """Combined loss with stronger regularization against collapse"""
    pos_src, pos_tgt = edge_index
    batch_size = pos_src.size(0)
    num_nodes = embeddings.size(0)
    
    # Positive samples (true edges)
    pos_scores = (embeddings[pos_src] * embeddings[pos_tgt]).sum(dim=1)
    
    # Hard negative mining: Sample nodes that are NOT connected but similar
    with torch.no_grad():
        # Create a candidate pool larger than needed
        candidate_nodes = torch.randint(0, num_nodes, (batch_size, neg_samples*2), device=embeddings.device)
        candidate_scores = []
        
        for i in range(neg_samples*2):
            candidates = candidate_nodes[:, i]
            # Ensure candidates are not in positive edges
            mask = (candidates == pos_tgt)
            candidates[mask] = (candidates[mask] + 1) % num_nodes
            
            # Calculate similarity to find hardest negatives
            candidate_score = (embeddings[pos_src] * embeddings[candidates]).sum(dim=1)
            candidate_scores.append(candidate_score)
        
        # Stack and find hardest negatives (highest similarity)
        candidate_scores = torch.stack(candidate_scores, dim=1)
        _, hard_indices = torch.topk(candidate_scores, neg_samples, dim=1)
        
        # Gather the hard negative nodes
        row_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, neg_samples)
        hard_neg_nodes = candidate_nodes[row_indices, hard_indices]
    
    # Compute loss with hard negatives
    neg_scores = []
    for i in range(neg_samples):
        neg_nodes = hard_neg_nodes[:, i]
        neg_score = (embeddings[pos_src] * embeddings[neg_nodes]).sum(dim=1)
        neg_scores.append(neg_score)
    
    neg_scores = torch.stack(neg_scores, dim=1)
    
    # InfoNCE loss (more effective than margin ranking)
    logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=embeddings.device)  # positive is at index 0
    
    # Cross entropy with temperature
    temperature = 0.1
    loss = F.cross_entropy(logits / temperature, labels)
    
    # Apply edge weight importance
    #weighted_loss = (edge_weight * loss).mean()
    
    # Add strong regularization against embedding collapse
    reg_loss = variance_regularization(embeddings)
    
    return  loss #weighted_loss #+ 1.0 * reg_loss  # Stronger regularization weight

def train_gat(cooc_matrix, id_dict, embedding_size, lr, num_epochs, dropout, batch_size, model_path, embeddings_init=None):
    graph_data = create_dataset(cooc_matrix, id_dict, embedding_size, embeddings_init)
    model = GAT(graph_data, embedding_size, embedding_size * 2, embedding_size, dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # Reduce LR gradually

    # Create edge dataset and DataLoader for batching
    edge_index = graph_data.edge_index.t()  # Convert to shape (num_edges, 2)
    edge_weight = graph_data.edge_attr

    dataset = TensorDataset(edge_index[:, 0], edge_index[:, 1], edge_weight)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print("Begin training with batched edges")
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0

        for batch in dataloader:
            optimizer.zero_grad()
            
            src_nodes, tgt_nodes, batch_weights = batch
            src_nodes, tgt_nodes, batch_weights = src_nodes.to(device), tgt_nodes.to(device), batch_weights.to(device)

            embeddings = model()
            batch_edge_index = torch.stack([src_nodes, tgt_nodes], dim=0)  # Shape (2, batch_size)

            # Compute loss on batched edges
            loss = loss_function(embeddings, batch_edge_index, batch_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f} sec", flush=True)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), model_path)
            save_model_embeddings(model, EMBEDDING_PATH, MODEL_NUM, EMBEDDING_SIZE, DATASET)
    
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')


def save_model_embeddings(model, EMBEDDING_PATH, MODEL_NUM, EMBEDDING_SIZE, DATASET):
    embeddings_array = model.get_embeddings()
    with open(f'{EMBEDDING_PATH}/gat_embeddings/gat_model_{MODEL_NUM}_emb_{EMBEDDING_SIZE}d_{export_name[DATASET]}.pickle', 'wb') as f:
        pickle.dump(embeddings_array, f)


if __name__ == "__main__":
    ## Load data
    with open(cooc_matrix_path[DATASET], 'rb') as file:
        cooc_matrix = pickle.load(file)
    with open(id_dict_path[DATASET], 'rb') as file:
        id_dict = pickle.load(file)
    with open(name_dict_path[DATASET], 'rb') as file:
        names_dict = pickle.load(file)


    inv_id_dict = {v: k for k, v in id_dict.items()}

    ## Train and save model
    if TRAIN_MODE:
        if INIT_EMBEDDING:
            print("Initialize embedding with Node2Vec")
            with open(init_node2vec_path[DATASET], 'rb') as f:
                embeddings_init = pickle.load(f)
            loss = train_gat(cooc_matrix, id_dict, EMBEDDING_SIZE, LR, NUM_EPOCHS, DROPOUT, BATCH_SIZE, model_path, embeddings_init)
        else:
            loss = train_gat(cooc_matrix, id_dict, EMBEDDING_SIZE, LR, NUM_EPOCHS, DROPOUT, BATCH_SIZE, model_path)

    ## Load model
    graph_data = create_dataset(cooc_matrix, id_dict, EMBEDDING_SIZE)
    model = GAT(graph_data, EMBEDDING_SIZE, EMBEDDING_SIZE * 2, EMBEDDING_SIZE, DROPOUT).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=False, map_location=device))

    save_model_embeddings(model, EMBEDDING_PATH, MODEL_NUM, EMBEDDING_SIZE, DATASET)