#!/gpfs/commons/home/aelhussein/anaconda3/envs/cuda_env_ne1/bin/python
#SBATCH --job-name=cui2vec_training
#SBATCH --partition=cpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aelhussein@nygenome.org
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --output=logs/outputs/output_cui2vec_model.txt
#SBATCH --error=logs/errors/errors_cui2vec_model.txt


## Choose model and specify iteration
"""
1: ROLLUP UKBB OMOP codes level 4 (UKBB)
2: ROLLUP UKBB OMOP codes level 5 (UKBB)
3: ROLLUP UKBB OMOP codes level 5 with minimum count filter (UKBB)
"""

## Imports
import os
os.chdir("/gpfs/commons/projects/ukbb-gursoylab/aelhussein")
import sys
sys.path.append("/gpfs/commons/projects/ukbb-gursoylab/aelhussein")
import pickle
import numpy as np
import sys
import pickle
import numpy as np
from scipy import sparse
from sklearn.utils.extmath import randomized_svd

DATASET = 4
MODEL_NUM = 1
TRAIN_MODE = True
EMBEDDING_SIZE  = 100
EMBEDDING_PATH = "trained_embeddings/cui2vec"
COOC_PATH = "datasets/cooc_matrices"
VOCAB_PATH = "datasets/vocab_dict"

# Params
model_name_dict = {1: 'ukbb_omop_rollup_lvl_4',
                   2: 'ukbb_omop_rollup_lvl_5',
                   3: 'ukbb_omop_rollup_lvl_5_ct_filter',
                   4: 'ukbb_omop_hierarchy_rollup_lvl_5_ct_filter'}

model_name = f"cui2vec_{MODEL_NUM}_emb_{EMBEDDING_SIZE}d_{model_name_dict[DATASET]}"

model_path = f'{EMBEDDING_PATH}/models/{model_name}.pkl'

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

export_name = {1: 'ukbb_omop_rollup_lvl_4',
               2: 'ukbb_omop_rollup_lvl_5',
               3: 'ukbb_omop_rollup_lvl_5_ct_filter',
               4: 'ukbb_omop_hierarchy_rollup_lvl_5_ct_filter'}

vocabulary = {1: 'omop',
              2: 'omop',
              3: 'omop',
              4: 'omop'}



class CUI2Vec:
    def __init__(self, embedding_size=100, smooth_factor=0.75, k=1):
        """
        Initialize CUI2Vec model
        
        Args:
            embedding_size: dimension of the embeddings
            smooth_factor: smoothing factor for PMI calculation
            k: shift factor for SPPMI
        """
        self.embedding_size = embedding_size
        self.smooth_factor = smooth_factor
        self.k = k

    def get_singleton_counts(self, cooccur_matrix):
        """
        Get singleton counts from cooccurrence matrix diagonal
        """
        if sparse.issparse(cooccur_matrix):
            singleton_counts = dict(enumerate(cooccur_matrix.diagonal()))
        else:
            singleton_counts = dict(enumerate(np.diag(cooccur_matrix)))
        return singleton_counts
        
    def construct_pmi(self, cooccur_matrix, N, epsilon=1e-6):
        """
        Construct PMI matrix from co-occurrence counts, handling divide-by-zero.
        """
        singleton_counts = self.get_singleton_counts(cooccur_matrix)
        
        # Convert to dense for processing if sparse
        if sparse.issparse(cooccur_matrix):
            cooccur_matrix = cooccur_matrix.todense()
        
        # Mask lower triangle since matrix is symmetric
        cooccur_matrix[np.tril_indices(cooccur_matrix.shape[0], -1)] = 0
        
        # Get concepts and their positions
        concepts = list(singleton_counts.keys())
        concept_to_idx = {c: i for i, c in enumerate(concepts)}
        
        # Calculate smoothed probabilities with epsilon to avoid divide-by-zero
        smoothed_counts = {cui: max((count / N) ** self.smooth_factor, epsilon) 
                        for cui, count in singleton_counts.items()}
        
        # Calculate PMI values
        rows, cols, values = [], [], []
        for i, c1 in enumerate(concepts):
            for j, c2 in enumerate(concepts[i:], i):
                if cooccur_matrix[i, j] > 0:
                    joint_prob = cooccur_matrix[i, j] / N
                    if joint_prob > 0:  # Only proceed if joint_prob is positive
                        pmi = np.log(joint_prob / 
                                    (smoothed_counts[c1] * smoothed_counts[c2]))
                        rows.append(i)
                        cols.append(j)
                        values.append(pmi)
                        
        return sparse.csr_matrix((values, (rows, cols)), 
                                shape=cooccur_matrix.shape)

    def construct_sppmi(self, pmi_matrix):
        """
        Construct Shifted Positive PMI matrix
        """
        # Shift and take positive values
        sppmi = pmi_matrix.copy()
        sppmi.data = np.maximum(sppmi.data - np.log(self.k), 0)
        
        # Remove zeros
        sppmi.eliminate_zeros()
        
        # Make symmetric
        sppmi = sppmi + sppmi.T
        
        return sppmi

    def train(self, cooccur_matrix, n_iter=25):
        """
        Train CUI2Vec model
        """
        if sparse.issparse(cooccur_matrix):
            N = cooccur_matrix.sum() // 2  # divide by 2 since matrix is symmetric
        else:
            N = np.sum(cooccur_matrix) // 2

        # Calculate PMI
        print("Calculating PMI matrix...", flush = True)
        pmi_matrix = self.construct_pmi(cooccur_matrix, N)
        
        # Calculate SPPMI
        print("Calculating SPPMI matrix...", flush = True)
        sppmi_matrix = self.construct_sppmi(pmi_matrix)
        
        # Perform SVD
        print(f"Performing SVD with {self.embedding_size} components...", flush = True)
        U, S, Vt = randomized_svd(sppmi_matrix, 
                                 n_components=self.embedding_size,
                                 n_iter=n_iter,
                                 random_state=42)
        
        # Create embeddings
        W = U @ np.diag(np.sqrt(S))
        C = Vt.T @ np.diag(np.sqrt(S))
        self.embeddings = W + C
        
        return self.embeddings

if __name__ == "__main__":    
    # Load data
    print("Loading data...")
    with open(cooc_matrix_path[DATASET], 'rb') as f:
        cooccur_matrix = pickle.load(f)
    
    
    # Initialize and train model
    model = CUI2Vec(embedding_size=EMBEDDING_SIZE)
    embeddings = model.train(cooccur_matrix)
    
    # Save embeddings
    print("Saving embeddings...", flush = True)
    output_path = f'{EMBEDDING_PATH}/cui2vec_embeddings/cui2vec_emb_{EMBEDDING_SIZE}d_{export_name[DATASET]}.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"Embeddings saved to {output_path}", flush = True)