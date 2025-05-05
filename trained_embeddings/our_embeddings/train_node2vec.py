#!/gpfs/commons/home/aelhussein/anaconda3/envs/cuda_env_ne1/bin/python
#SBATCH --job-name=node2vec_training
#SBATCH --partition=cpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aelhussein@nygenome.org
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=logs/outputs/output_node2vec.txt
#SBATCH --error=logs/errors/errors_node2vec.txt

import os
os.chdir("/gpfs/commons/projects/ukbb-gursoylab/aelhussein")
import sys
sys.path.append("/gpfs/commons/projects/ukbb-gursoylab/aelhussein")
import pickle
from node2vec import Node2Vec
import numpy as np
import pandas as pd
import os
import networkx as nx

"""
1: ROLLUP UKBB OMOP codes level 4 
2: ROLLUP UKBB OMOP codes level 5
3: ROLLUP UKBB OMOP codes level 5 with minimum count filter (UKBB)
"""
MODEL_NUM = 1
ROLLUP_LVL = 5
DATASET = 3
TREE = "OMOP" # ICD, OMOP_ICD or OMOP
EMBEDDING_SIZE = 100
WALK_LENGTH = 30
NUM_WALKS = 750

VOCAB_PATH = "datasets/vocab_dict"
HIERARCHY_PATH = "datasets/code_hierarchies"
EMBEDDING_PATH = "trained_embeddings/our_embeddings"

id_dict_path = {1: f'{VOCAB_PATH}/code2id_ukbb_omop_rollup_lvl_4.pickle',
                2: f'{VOCAB_PATH}/code2id_ukbb_omop_rollup_lvl_5.pickle',
                3: f'{VOCAB_PATH}/code2id_ukbb_omop_rollup_lvl_5_ct_filter.pickle',}

name_dict_path = {1: f'{VOCAB_PATH}/code2name_ukbb_omop_rollup_lvl_4.pickle', 
                  2: f'{VOCAB_PATH}/code2name_ukbb_omop_rollup_lvl_5.pickle',
                  3: f'{VOCAB_PATH}/code2name_ukbb_omop_rollup_lvl_5_ct_filter.pickle', }

export_name = {1: 'ukbb_omop_rollup_lvl_4',
               2: 'ukbb_omop_rollup_lvl_5',
               3: 'ukbb_omop_rollup_lvl_5_ct_filter'}

vocabulary = {1: 'omop',
              2: 'omop',
              3: 'omop'}


model_name = f"node2vec_{MODEL_NUM}_{TREE}_emb_{EMBEDDING_SIZE}d_{export_name[DATASET]}"


def inverse_dict(d):
    inv_d = {}
    for k, v_list in d.items():
        for v in v_list:
            inv_d[v] = k
    return inv_d


def get_vector_iso(code, node_embeddings):
    """
    Return ICD node2vec embedding for ICD vector or return OMOP node2vec embedding for OMOP vector.
    """
    try :
        return node_embeddings.get_vector(list(np.array(node_embeddings.index_to_key).astype(int)).index(code))
    except ValueError:
        print(code)
        return np.mean(node_embeddings.vectors, axis=0)
    
def build_index_mapping(node_embeddings):
    """
    Build a dictionary to map code to the index in node_embeddings.
    """
    return {int(key): i for i, key in enumerate(node_embeddings.index_to_key)}

def get_vector_iso(code, node_embeddings, index_mapping, mean_vector):
    """
    Return ICD/OMOP embedding for the given code or the mean vector if not found.
    """
    index = index_mapping.get(code)
    if index is not None:
        return node_embeddings.get_vector(index)
    else:
        print(f"Code {code} not found, returning mean vector.")
        return mean_vector


if __name__ == "__main__":

    if TREE == "OMOP":
        with open(f'{HIERARCHY_PATH}/ukbb_omop_tree_filtered.pickle', 'rb') as f:
            G = pickle.load(f)

    with open(f'{id_dict_path[DATASET]}', 'rb') as file:
        id_dict = pickle.load(file)
    with open(f'{name_dict_path[DATASET]}', 'rb') as file:
        names_dict = pickle.load(file)
    inv_id_dict = {v: k for k, v in id_dict.items()}
    
    node2vec = Node2Vec(G, dimensions=EMBEDDING_SIZE, 
                        walk_length=WALK_LENGTH, num_walks=NUM_WALKS, p=1, q=1,
                        workers=4)

    model = node2vec.fit(window=10, min_count=1, batch_words=4096)
    print('Node2Vec fit', flush = True)

    with open(f'{EMBEDDING_PATH}/models/node2vec_model_{MODEL_NUM}_{EMBEDDING_SIZE}_rollup_lvl_{ROLLUP_LVL}.pickle', 'wb') as f:
        pickle.dump(model, f)
    print('Node2Vec model saved', flush = True)

    if vocabulary[DATASET]=='icd':
        pass
    elif vocabulary[DATASET]=='omop':
        if TREE == "OMOP":
            keys = list(id_dict.keys())
            node_embeddings = model.wv
            index_mapping = build_index_mapping(node_embeddings)
            mean_vector = np.mean(node_embeddings.vectors, axis=0)
            vectors = [get_vector_iso(key, node_embeddings, index_mapping, mean_vector) for key in keys]
            matrix = np.vstack(vectors)

    with open(f'{EMBEDDING_PATH}/node2vec_embeddings/n2v_model_{MODEL_NUM}_emb_{EMBEDDING_SIZE}d_{export_name[DATASET]}.pickle', 'wb') as f:
        pickle.dump(matrix, f)
    print('Node2Vec embeddings saved', flush = True)


