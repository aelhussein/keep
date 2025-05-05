#!/gpfs/commons/home/aelhussein/anaconda3/envs/cuda_env_ne1/bin/python
#SBATCH --job-name=pretrained_ext
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aelhussein@nygenome.org
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/outputs/output_pretrained.txt
#SBATCH --error=logs/errors/errors_pretrained.txt

from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, BioGptTokenizer, BioGptForCausalLM
import torch
import pandas as pd
import numpy as np
import pickle
import networkx as nx
from collections import defaultdict
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, flush=True)

UKBB_DATA_DIR = "/gpfs/commons/datasets/controlled/ukbb-gursoylab"
ROOT_DIR = "/gpfs/commons/projects/ukbb-gursoylab/aelhussein"
rollup_lvl = 5
dataset = '_ct_filter'


def mean_pooling(last_hidden_state, attention_mask):
    ''' For GPT style models, get the mean embedding'''
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask

def extract_embeddings(omop_codes, tokenizer, model, omop_embedding, model_type):
    ''' Extracts embeddings
        1. CLS token for BERT
        2. Mean pooling for GPT
    '''
    model.to(device)
    for code, description in omop_codes.items():
        inputs = tokenizer(description, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1]
        if model_type == 'BERT': #cls
            embedding = last_hidden_state[:, 0, :]
        elif model_type == 'GPT': #mean pooling
            attention_mask = inputs['attention_mask'].to(device)
            embedding = mean_pooling(last_hidden_state, attention_mask)
        omop_embedding[code] = embedding.cpu().squeeze().numpy()
    model.to('cpu')
    return

def get_node_ancestry_descriptions(G, code_name_dict):
    """
    For each node, get its description and the descriptions of its parents and grandparents,
    grouping by parent when there are multiple grandparents.
    """
    def get_parents(node):
        """Get immediate parents of a node."""
        return set(G.predecessors(node))
    
    def get_grandparents(parent):
        """Get parents of a parent node (grandparents)."""
        return set(G.predecessors(parent))
    
    ancestry_descriptions = {}
    
    for node in G.nodes():
        node_desc = code_name_dict.get(node)
        if not node_desc:  # Skip nodes without descriptions
            continue
            
        parents = get_parents(node)
        if not parents:
            ancestry_descriptions[node] = node_desc
            continue
        
        # Group grandparents by parent
        parent_grandparent_map = defaultdict(set)
        for parent in parents:
            parent_desc = code_name_dict.get(parent)
            if not parent_desc:
                continue
                
            grandparents = get_grandparents(parent)
            valid_grandparents = set()
            
            for grandparent in grandparents:
                grandparent_desc = code_name_dict.get(grandparent)
                if grandparent_desc:
                    valid_grandparents.add(grandparent_desc)
            
            if valid_grandparents:
                parent_grandparent_map[parent_desc].update(valid_grandparents)
        
        # Build ancestry paths
        ancestry_paths = []
        
        # Handle parents with grandparents
        for parent_desc, grandparent_descs in parent_grandparent_map.items():
            if grandparent_descs:
                grandparents_str = " and ".join(sorted(grandparent_descs))
                ancestry_paths.append(f"is a {parent_desc} is a {grandparents_str}")
            else:
                ancestry_paths.append(f"is a {parent_desc}")
        
        # Handle parents without any valid grandparents
        for parent in parents:
            parent_desc = code_name_dict.get(parent)
            if parent_desc and parent_desc not in parent_grandparent_map:
                ancestry_paths.append(f"is a {parent_desc}")
        
        # Combine all paths
        if ancestry_paths:
            ancestry_str = f"{node_desc}; " + "; ".join(sorted(ancestry_paths))
        else:
            ancestry_str = node_desc
            
        ancestry_descriptions[node] = ancestry_str
    
    return ancestry_descriptions

def dict_to_indexed_array(embedding_dict):
    max_key = max(embedding_dict.keys())
    embedding_dim = len(next(iter(embedding_dict.values())))
    indexed_array = np.zeros((max_key + 1, embedding_dim))
    for key, vector in embedding_dict.items():
        indexed_array[key] = vector
    return indexed_array


def main():
    # Load models
    bio_clin_bert_tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    bio_clin_bert_model = BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', output_hidden_states=True)

    clin_bert_tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    clin_bert_model = AutoModel.from_pretrained("medicalai/ClinicalBERT", output_hidden_states=True)

    biogpt_tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    biogpt_model = BioGptForCausalLM.from_pretrained("microsoft/biogpt", output_hidden_states=True)

    # Load code dictionaries    
    with open(f'{ROOT_DIR}/datasets/vocab_dict/code2id_ukbb_omop_rollup_lvl_{rollup_lvl}{dataset}.pickle', 'rb') as f:
        code_id_dict = pickle.load(f)

    with open(f'{ROOT_DIR}/datasets/vocab_dict/code2name_ukbb_omop_rollup_lvl_{rollup_lvl}{dataset}.pickle', 'rb') as f:
        code_name_dict = pickle.load(f)
    
    id_name_dict = {code_id_dict[id]:code_name_dict[id] for id in code_id_dict.keys()}
    

    # Loop through models and get embeddings
    tokenizers = [bio_clin_bert_tokenizer, clin_bert_tokenizer, biogpt_tokenizer]
    models = [bio_clin_bert_model, clin_bert_model, biogpt_model]
    bio_clin_bert_embeddings = {}; clin_bert_embeddings = {}; biogpt_embeddings = {}
    omop_embeddings = [bio_clin_bert_embeddings, clin_bert_embeddings, biogpt_embeddings]
    model_types = ["BERT", "BERT", "GPT"]
    for tokenizer, model, omop_embedding, model_type in zip(tokenizers, models, omop_embeddings, model_types):
        extract_embeddings(id_name_dict, tokenizer, model, omop_embedding, model_type)

    print('Basic description completed', flush = True)
    
    # Save Models
    bio_clin_bert_arr = dict_to_indexed_array(bio_clin_bert_embeddings)
    clin_bert_arr = dict_to_indexed_array(clin_bert_embeddings)
    biogpt_arr = dict_to_indexed_array(biogpt_embeddings)
    with open(f'{ROOT_DIR}/pretrained_embeddings/bio_clin_bert_embeddings_rollup_lvl_{rollup_lvl}{dataset}.pkl', 'wb') as f:
        pickle.dump(bio_clin_bert_arr, f)

    with open(f'{ROOT_DIR}/pretrained_embeddings/clin_bert_embeddings_rollup_lvl_{rollup_lvl}{dataset}.pkl', 'wb') as f:
        pickle.dump(clin_bert_arr, f)

    with open(f'{ROOT_DIR}/pretrained_embeddings/biogpt_embeddings_rollup_lvl_{rollup_lvl}{dataset}.pkl', 'wb') as f:
        pickle.dump(biogpt_arr, f)

    ### Hierarchical aware ###

    # Load graph and additional concept names
    concept = pd.read_csv(f"{ROOT_DIR}/datasets/code_hierarchies/CONCEPT.csv", sep = '\t', low_memory = False)
    G = pickle.load(open(f'{ROOT_DIR}/datasets/code_hierarchies/ukbb_omop_tree_filtered.pickle', 'rb'))
    code_name_dict_all = concept.set_index('concept_id')[['concept_name']].to_dict()['concept_name']
    code_name_dict_all.update(code_name_dict) 

    # Get the ancestry descriptions
    ancestry_dict = get_node_ancestry_descriptions(G, code_name_dict_all)

    # Create index id: description dictionary
    id_name_dict_hierarchy = {code_id_dict[id]:ancestry_dict[id] for id in code_id_dict.keys()}

    # Loop through models with hierarchical descriptions and get embeddings
    tokenizers = [bio_clin_bert_tokenizer, clin_bert_tokenizer, biogpt_tokenizer]
    models = [bio_clin_bert_model, clin_bert_model, biogpt_model]
    bio_clin_bert_embeddings_hier = {}; clin_bert_embeddings_hier = {}; biogpt_embeddings_hier = {}
    omop_embeddings = [bio_clin_bert_embeddings_hier, clin_bert_embeddings_hier, biogpt_embeddings_hier]
    model_types = ["BERT", "BERT", "GPT"]

    for tokenizer, model, omop_embedding, model_type in zip(tokenizers, models, omop_embeddings, model_types):
        extract_embeddings(id_name_dict_hierarchy, tokenizer, model, omop_embedding, model_type)

    print('Hierarchical description completed', flush = True)
    # Save hierarchical embeddings
    bio_clin_bert_hier_arr = dict_to_indexed_array(bio_clin_bert_embeddings_hier)
    clin_bert_hier_arr = dict_to_indexed_array(clin_bert_embeddings_hier)
    biogpt_hier_arr = dict_to_indexed_array(biogpt_embeddings_hier)

    with open(f'{ROOT_DIR}/pretrained_embeddings/bio_clin_bert_embeddings_rollup_lvl_{rollup_lvl}_hierarchy{dataset}.pkl', 'wb') as f:
        pickle.dump(bio_clin_bert_hier_arr, f)

    with open(f'{ROOT_DIR}/pretrained_embeddings/clin_bert_embeddings_rollup_lvl_{rollup_lvl}_hierarchy{dataset}.pkl', 'wb') as f:
        pickle.dump(clin_bert_hier_arr, f)

    with open(f'{ROOT_DIR}/pretrained_embeddings/biogpt_embeddings_rollup_lvl_{rollup_lvl}_hierarchy{dataset}.pkl', 'wb') as f:
        pickle.dump(biogpt_hier_arr, f)


if __name__ == "__main__":
    main()