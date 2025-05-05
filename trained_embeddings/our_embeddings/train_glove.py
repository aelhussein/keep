#!/gpfs/commons/home/aelhussein/anaconda3/envs/cuda_env_ne1/bin/python
#SBATCH --job-name=glove_training
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aelhussein@nygenome.org
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/outputs/output_glove_model_reg_1e-5.txt
#SBATCH --error=logs/errors/errors_glove_model_reg_1e-5.txt
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
REGULARIZATION = True  # if True INIT_EMBEDDING must be True too

SPARSE = False # if True --> replace 1 values with 0 in coocurrence matrix
REG_NORM = None # if None cosine distance will be used for regularization else it will be norm p with p = REG_NORM

## Hyperparameters
ALPHA = 0.75
X_MAX = 50
x_max_quantile = 0.75
EMBEDDING_SIZE = 100
LR = 0.05
NUM_EPOCHS = 300
BATCH_SIZE = 1024
LAMBD = 0.00001

EMBEDDING_PATH = "trained_embeddings/our_embeddings"
COOC_PATH = "datasets/cooc_matrices"
VOCAB_PATH = "datasets/vocab_dict"

# Params
model_name_dict = {1: 'ukbb_omop_rollup_lvl_4',
                   2: 'ukbb_omop_rollup_lvl_5',
                   3: 'ukbb_omop_rollup_lvl_5_ct_filter'}

if REGULARIZATION:
    model_name = f"glove_model_{MODEL_NUM}_emb_{EMBEDDING_SIZE}d_{model_name_dict[DATASET]}_REG_{LAMBD}"
else:
    model_name = f"glove_model_{MODEL_NUM}_emb_{EMBEDDING_SIZE}d_{model_name_dict[DATASET]}"

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


init_node2vec_path = {1: f'{EMBEDDING_PATH}/node2vec_embeddings/n2v_model_{MODEL_NUM_N2V}_emb_{EMBEDDING_SIZE}d_ukbb_omop_rollup_lvl_4.pickle',
                      2: f'{EMBEDDING_PATH}/node2vec_embeddings/n2v_model_{MODEL_NUM_N2V}_emb_{EMBEDDING_SIZE}d_ukbb_omop_rollup_lvl_5.pickle',
                      3: f'{EMBEDDING_PATH}/node2vec_embeddings/n2v_model_{MODEL_NUM_N2V}_emb_{EMBEDDING_SIZE}d_ukbb_omop_rollup_lvl_5_ct_filter.pickle',
                      4: f'{EMBEDDING_PATH}/node2vec_embeddings/n2v_model_{MODEL_NUM_N2V}_emb_{EMBEDDING_SIZE}d_ukbb_omop_rollup_lvl_5_ct_filter.pickle'}

export_name = {1: 'ukbb_omop_rollup_lvl_4',
               2: 'ukbb_omop_rollup_lvl_5',
               3: 'ukbb_omop_rollup_lvl_5_ct_filter',
               4: 'ukbb_omop_hierarchy_rollup_lvl_5_ct_filter'}

vocabulary = {1: 'omop',
              2: 'omop',
              3: 'omop',
              4: 'omop'}


## Glove Model
class GloveDataset(Dataset):
    def __init__(self, cooc_matrix, num_words, x_max, alpha):
        super(GloveDataset, self).__init__()
        self.data = []
        for i in range(cooc_matrix.shape[0]):
            for j in range(cooc_matrix.shape[1]):
                if cooc_matrix[i, j] > 0:
                    self.data.append((i, j, cooc_matrix[i, j]))
        self.cooc_matrix = cooc_matrix
        self.num_words = num_words
        self.x_max = x_max
        self.alpha = alpha

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        i, j, count = self.data[idx]
        weight = (count / self.x_max) ** self.alpha if count < self.x_max else 1.0
        return torch.tensor(i), torch.tensor(j), torch.tensor(count).float(), torch.tensor(weight).float()


class Glove(nn.Module):
    def __init__(self, num_words, embedding_size, embeddings_init=None, lambd = None, reg_norm=None, log = False):
        super(Glove, self).__init__()
        self.embeddings_v = torch.nn.Embedding(num_words, embedding_size)
        self.embeddings_u = torch.nn.Embedding(num_words, embedding_size)
        self.biases_u = torch.nn.Embedding(num_words, 1)
        self.biases_v = torch.nn.Embedding(num_words, 1)
        if embeddings_init is not None:
            self.embeddings_v.weight.data.copy_(torch.from_numpy(embeddings_init))
            self.embeddings_u.weight.data.copy_(torch.from_numpy(embeddings_init))
        else:
            self.embeddings_v.weight.data.uniform_(-0.5, 0.5)
            self.embeddings_u.weight.data.uniform_(-0.5, 0.5)
        self.initial_embeddings = self.embeddings_u.weight.data.clone().to(device)
        self.biases_v.weight.data.fill_(0)
        self.biases_u.weight.data.fill_(0)
        self.lambd = lambd
        self.regularization = embeddings_init is not None and lambd is not None # No regularization if no initial embedding is provided
        self.reg_norm = reg_norm
        self.log = log

    def forward(self, i_indices, j_indices, counts, weights):
        embedding_i = self.embeddings_v(i_indices)
        embedding_j = self.embeddings_u(j_indices)
        bias_i = self.biases_v(i_indices).squeeze()
        bias_j = self.biases_u(j_indices).squeeze()
        dot_product = torch.sum(embedding_i * embedding_j, dim=1)
        loss = weights * ((dot_product + bias_i + bias_j - torch.log(counts)) ** 2)
        if self.regularization:
            u_plus_v_i = (embedding_i + self.embeddings_u(i_indices))/2
            u_plus_v_j = (embedding_j + self.embeddings_v(j_indices))/2
            if self.reg_norm is None:
                reg_dist_i = 1 - torch.nn.functional.cosine_similarity(u_plus_v_i, self.initial_embeddings[i_indices], dim=1) 
                reg_dist_j = 1 - torch.nn.functional.cosine_similarity(u_plus_v_j, self.initial_embeddings[j_indices], dim=1)
            else:
                reg_dist_i = torch.norm(u_plus_v_i - self.initial_embeddings[i_indices], p=self.reg_norm, dim=1)
                reg_dist_j = torch.norm(u_plus_v_j - self.initial_embeddings[j_indices], p=self.reg_norm, dim=1)
            if self.log:
                reg_dist_i = torch.log(reg_dist_i + 1e-5)
                reg_dist_j = torch.log(reg_dist_j + 1e-5)
            #print(f'Shape or reg is {reg_dist_i.shape}')
            reg_loss = self.lambd * (torch.sum(reg_dist_i) + torch.sum(reg_dist_j))
            loss += reg_loss
            return loss, reg_loss
        return loss, torch.zeros_like(loss).to(device)


def train_glove(cooc_matrix, embedding_size, lr, num_epochs, batch_size, x_max, alpha, model_path, embeddings_init=None, lambd = None, reg_norm=None):
    num_words = cooc_matrix.shape[0]

    glove_dataset = GloveDataset(cooc_matrix, num_words, x_max, alpha)
    data_loader = DataLoader(glove_dataset, batch_size=batch_size, shuffle=True)

    model = Glove(num_words, embedding_size, embeddings_init, lambd, reg_norm)
    model.to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only = True))

    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

    print("Begin training")

    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0
        total_reg_loss = 0
        for i_indices, j_indices, counts, weights in data_loader:
            i_indices = i_indices.to(device)
            j_indices = j_indices.to(device)
            counts = counts.to(device)
            weights = weights.to(device)

            optimizer.zero_grad()
            loss, reg_loss = model(i_indices, j_indices, counts, weights)
            loss.mean().backward()
            optimizer.step()

            total_loss += loss.mean().item()
            total_reg_loss += reg_loss.mean().item()

        epoch_time = time.time() - start_time
        
        if REGULARIZATION:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader):.4f}, Reg Loss: {total_reg_loss/len(data_loader):.4f}, time: {epoch_time:.2f} seconds", flush = True)
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader):.4f}, time: {epoch_time:.2f} seconds", flush = True)
        
        if epoch % 50 == 0:
            ## Save model and embeddings
            torch.save(model.state_dict(), model_path)
            if REGULARIZATION:
                save_model_embeddings(model, EMBEDDING_PATH, MODEL_NUM, EMBEDDING_SIZE, DATASET, LAMBD)
            else:
                save_model_embeddings(model, EMBEDDING_PATH, MODEL_NUM, EMBEDDING_SIZE, DATASET)
    
    torch.save(model.state_dict(), model_path)
    print('Model saved to %s' % model_path)

    return total_loss/len(data_loader)

def load_glove_embeddings(model):
    embeddings_u = model.embeddings_u.weight.data.cpu().data.numpy()  
    embeddings_v = model.embeddings_v.weight.data.cpu().data.numpy() 
    combined_embeddings = (embeddings_u + embeddings_v) / 2
    return combined_embeddings


def save_model_embeddings(model, EMBEDDING_PATH, MODEL_NUM, EMBEDDING_SIZE, DATASET, LAMBD = None):
        embeddings_array = load_glove_embeddings(model)
        if REGULARIZATION:
            with open(f'{EMBEDDING_PATH}/glove_embeddings/glove_model_{MODEL_NUM}_emb_{EMBEDDING_SIZE}d_{export_name[DATASET]}_REG_{LAMBD}.pickle', 'wb') as f:
                pickle.dump(embeddings_array, f)
        else:
            with open(f'{EMBEDDING_PATH}/glove_embeddings/glove_model_{MODEL_NUM}_emb_{EMBEDDING_SIZE}d_{export_name[DATASET]}.pickle', 'wb') as f:
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


    X_MAX = max(X_MAX, np.quantile(cooc_matrix.ravel(), x_max_quantile)) ## adapt x max to the coocurrence svalues
    print(f"X_max: {X_MAX}")

    # Remove 1 values in coocurrence matrix
    if SPARSE:
        cooc_matrix = np.where(cooc_matrix == 1, 0, cooc_matrix)


    ## Train and save model
    if TRAIN_MODE:
        if INIT_EMBEDDING:
            print("Initialize embedding")
            with open(init_node2vec_path[DATASET], 'rb') as f:
                embeddings_init = pickle.load(f)
            if REGULARIZATION:
                loss = train_glove(cooc_matrix, EMBEDDING_SIZE, LR, NUM_EPOCHS, BATCH_SIZE, X_MAX, ALPHA, model_path, embeddings_init, LAMBD, REG_NORM)
            else:
                loss = train_glove(cooc_matrix, EMBEDDING_SIZE, LR, NUM_EPOCHS, BATCH_SIZE, X_MAX, ALPHA, model_path, embeddings_init)
        else:
            loss = train_glove(cooc_matrix, EMBEDDING_SIZE, LR, NUM_EPOCHS, BATCH_SIZE, X_MAX, ALPHA, model_path)


    ## Load model
    model = Glove(cooc_matrix.shape[0], EMBEDDING_SIZE)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    ## Save model
    if REGULARIZATION:
        save_model_embeddings(model, EMBEDDING_PATH, MODEL_NUM, EMBEDDING_SIZE, DATASET, LAMBD)
    else:
        save_model_embeddings(model, EMBEDDING_PATH, MODEL_NUM, EMBEDDING_SIZE, DATASET)
