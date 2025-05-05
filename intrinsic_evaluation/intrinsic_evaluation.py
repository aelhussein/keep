#!/gpfs/commons/home/aelhussein/anaconda3/envs/cuda_env_ne1/bin/python
#SBATCH --job-name=intrinsic_evaluation
#SBATCH --partition=cpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aelhussein@nygenome.org
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --output=logs/outputs/output_eval_1.txt
#SBATCH --error=logs/errors/errors_eval_1.txt

#DIRECTORIES
ROOT_DIR = '/gpfs/commons/projects/ukbb-gursoylab'
WORKING_DIR = f'{ROOT_DIR}/aelhussein'
UKBB_DATA_DIR = "/gpfs/commons/datasets/controlled/ukbb-gursoylab"

#IMPORTS
import sys
sys.path.append("/gpfs/commons/projects/ukbb-gursoylab/aelhussein")
import pandas as pd
import numpy as np
import pickle
import torch
import importlib
import utils
from scipy.stats import rankdata, wilcoxon
from utils import semantic_sim_correlation, cooccurrence_sim_correlation, evaluate_known_relationships

#PARAMS
rollup_lvl = 5
dataset = 1
hierarchy = ''
RUNS = 250
K1 = 10
K2 = 150

#PATHS
code_dict_path = {1: f'{WORKING_DIR}/datasets/vocab_dict/code2id_ukbb_omop_rollup_lvl_{rollup_lvl}_ct_filter.pickle'}
name_dict_path = {1: f'{WORKING_DIR}/datasets/vocab_dict/code2name_ukbb_omop_rollup_lvl_{rollup_lvl}_ct_filter.pickle'}
omop_tree_path = {1: f'{WORKING_DIR}/datasets/code_hierarchies/ukbb_omop_tree_filtered.pickle'}
cooc_matrix_path = {1: f"{WORKING_DIR}/datasets/cooc_matrices/cooc_ukbb_omop_hierarchy_rollup_lvl_{rollup_lvl}_ct_filter.pickle"}
node_similarity_path = {1: f'{WORKING_DIR}/datasets/code_hierarchies/node_similarity_scores_filtered.pkl'}
embedding_path = {1: '_ct_filter'}

#DICTIONARIES AND TREES
with open(code_dict_path[dataset], 'rb') as f:
    code_dict = pickle.load(f)
inv_code_dict = {v: k for k, v in code_dict.items()}
codes = list(code_dict.keys())

with open(name_dict_path[dataset], 'rb') as f:
    name_dict = pickle.load(f)
name_id_dict = {code_dict[k]:name_dict[k] for k in code_dict.keys()}
inv_name_id_dict = {v:k for k,v in name_id_dict.items()}

with open(omop_tree_path[dataset], 'rb') as f:
    omop_tree = pickle.load(f)

with open(node_similarity_path[dataset], 'rb') as f:
    node_similarity = pickle.load(f)

def load_embeddings(rollup_lvl, dataset):
    """ Load embeddings """
    
    with open(f'{WORKING_DIR}/pretrained_embeddings/bio_clin_bert_embeddings_rollup_lvl_{rollup_lvl}{embedding_path[dataset]}.pkl', 'rb') as f:
        bio_clin_bert = pickle.load(f)

    with open(f'{WORKING_DIR}/pretrained_embeddings/clin_bert_embeddings_rollup_lvl_{rollup_lvl}{embedding_path[dataset]}.pkl', 'rb') as f:
        clin_bert = pickle.load(f)

    with open(f'{WORKING_DIR}/pretrained_embeddings/biogpt_embeddings_rollup_lvl_{rollup_lvl}{embedding_path[dataset]}.pkl', 'rb') as f:
        biogpt = pickle.load(f)

    with open(f'{WORKING_DIR}/pretrained_embeddings/bio_clin_bert_embeddings_rollup_lvl_{rollup_lvl}_hierarchy{embedding_path[dataset]}.pkl', 'rb') as f:
        bio_clin_bert_hier = pickle.load(f)

    with open(f'{WORKING_DIR}/pretrained_embeddings/clin_bert_embeddings_rollup_lvl_{rollup_lvl}_hierarchy{embedding_path[dataset]}.pkl', 'rb') as f:
        clin_bert_hier = pickle.load(f)

    with open(f'{WORKING_DIR}/pretrained_embeddings/biogpt_embeddings_rollup_lvl_{rollup_lvl}_hierarchy{embedding_path[dataset]}.pkl', 'rb') as f:
        biogpt_hier = pickle.load(f)

    with open(f'{WORKING_DIR}/trained_embeddings/our_embeddings/node2vec_embeddings/n2v_model_1_emb_100d_ukbb_omop_rollup_lvl_{rollup_lvl}{embedding_path[dataset]}.pickle', 'rb') as f:
        n2v = pickle.load(f)

    with open(f'{WORKING_DIR}/trained_embeddings/our_embeddings/glove_embeddings/glove_model_1_emb_100d_ukbb_omop_rollup_lvl_{rollup_lvl}{embedding_path[dataset]}.pickle', 'rb') as f:
        glove = pickle.load(f)

    with open(f'{WORKING_DIR}/trained_embeddings/our_embeddings/glove_embeddings/glove_model_1_emb_100d_ukbb_omop_rollup_lvl_{rollup_lvl}{embedding_path[dataset]}_REG_0.0001.pickle', 'rb') as f:
        glove_n2v = pickle.load(f)

    with open(f'{WORKING_DIR}/trained_embeddings/cui2vec/cui2vec_embeddings/cui2vec_emb_100d_ukbb_omop_hierarchy_rollup_lvl_{rollup_lvl}{embedding_path[dataset]}.pkl', 'rb') as f:
        cui2vec = pickle.load(f)

    return bio_clin_bert, clin_bert, biogpt, bio_clin_bert_hier, clin_bert_hier, biogpt_hier, n2v, glove, glove_n2v, cui2vec

def load_convert_tensor(rollup_lvl, dataset):
    """ Convert embeddings to tensor """
    bio_clin_bert, clin_bert, biogpt, bio_clin_bert_hier, clin_bert_hier, biogpt_hier, n2v, glove, glove_n2v, cui2vec = load_embeddings(rollup_lvl, dataset)
    bio_clin_bert_tensor = torch.tensor(bio_clin_bert)
    clin_bert_tensor = torch.tensor(clin_bert)
    biogpt_tensor = torch.tensor(biogpt)
    bio_clin_bert_hier_tensor = torch.tensor(bio_clin_bert_hier)
    clin_bert_hier_tensor = torch.tensor(clin_bert_hier)
    biogpt_hier_tensor = torch.tensor(biogpt_hier)
    n2v_tensor  = torch.tensor(n2v)
    glove_tensor  = torch.tensor(glove)
    glove_n2v_tensor  = torch.tensor(glove_n2v)
    cui2vec_tensor = torch.tensor(cui2vec)

    embedding_tensors = {
    'BioClinBERT': bio_clin_bert_tensor,
    'ClinBERT': clin_bert_tensor,
    'BioGPT': biogpt_tensor,
    'BioClinBERT_hier': bio_clin_bert_hier_tensor,
    'ClinBERT_hier': clin_bert_hier_tensor,
    'BioGPT_hier': biogpt_hier_tensor,
    'node2vec': n2v_tensor,
    'GloVe': glove_tensor,
    'GloVe+node2vec': glove_n2v_tensor,
    'Cui2vec': cui2vec_tensor
    }

    return embedding_tensors


def node_similarities(rollup_lvl, dataset):
    """ Tree-based similarities """
    embedding_tensors = load_convert_tensor(rollup_lvl, dataset)
    resnik_corr_runs = {model:[] for model in embedding_tensors.keys()}
    lin_corr_runs = {model:[] for model in embedding_tensors.keys()}
    for run in range(RUNS):
        for model, embedding_tensor in embedding_tensors.items():
            # Calculate correlations for both Resnik and Lin similarities
            resnik_corr = semantic_sim_correlation(semantic_similarities=node_similarity,
                                                embedding_tensor=embedding_tensor,
                                                cats=codes,
                                                K1=K1,  
                                                K2=K2,
                                                code_dict=code_dict,
                                                inv_code_dict=inv_code_dict,
                                                similarity_type='resnik')
            resnik_corr_runs[model].append(resnik_corr)
            lin_corr = semantic_sim_correlation(semantic_similarities=node_similarity,
                                            embedding_tensor=embedding_tensor,
                                            cats=codes,
                                            K1=K1,
                                            K2=K2,
                                            code_dict=code_dict,
                                            inv_code_dict=inv_code_dict,
                                            similarity_type='lin') 
            lin_corr_runs[model].append(lin_corr)
        
    # Save results
    with open(f'{WORKING_DIR}/intrinsic_evaluation/results/resnik_correlation_results_rollup_lvl_{rollup_lvl}{embedding_path[dataset]}.pkl', 'wb') as f:
        pickle.dump(resnik_corr_runs, f)
    
    with open(f'{WORKING_DIR}/intrinsic_evaluation/results/lin_correlation_results_rollup_lvl_{rollup_lvl}{embedding_path[dataset]}.pkl', 'wb') as f:
        pickle.dump(lin_corr_runs, f)

    return


## COOCCURENCE BASED SIMILARITY
def cooc_similarities(rollup_lvl, dataset):
    """ COOC based similarity """
    with open(cooc_matrix_path[dataset], 'rb') as file:
            cooc_matrix = pickle.load(file)
     
    embedding_tensors = load_convert_tensor(rollup_lvl, dataset)

    cooccurrence_results = {model:[] for model in embedding_tensors.keys()}

    # Calculate correlations for each embedding model
    for run in range(RUNS):
        for model_name, embedding_tensor in embedding_tensors.items():
            correlation = cooccurrence_sim_correlation(cooc_matrix,embedding_tensor,codes,K1,K2,code_dict,inv_code_dict)
            cooccurrence_results[model_name].append(correlation)

    # Save results
    with open(f'{WORKING_DIR}/intrinsic_evaluation/results/cooccurrence_correlation_results_rollup_lvl_{rollup_lvl}{embedding_path[dataset]}.pkl', 'wb') as f:
        pickle.dump(cooccurrence_results, f)
    
    return 

# COMORBIDITY BENCHMARK
def comorb_similarities(rollup_lvl, dataset):
    """ COMORB SIMIALRITY LIKE BEAM ET AL. """
    embedding_tensors = load_convert_tensor(rollup_lvl, dataset)

    ## Clinical Concept Embeddings Learned from Massive Sources of Multimodal Medical Data
    core_concepts = ['Asthma',
                    'Obesity',
                    'Premature rupture of membranes',
                    'Renal transplant rejection',
                    'Schizophrenia',
                    'Tumor of respiratory system',
                    'Type 1 diabetes mellitus',
                    'Type 2 diabetes mellitus',
                    ]
    power_results = {(core_concept, model): [] for core_concept in core_concepts for model in embedding_tensors.keys()}

    for run in range(RUNS):
        for core_concept in core_concepts:
            core_concept_filename = core_concept.replace(" ", "_")
            with open(f'{WORKING_DIR}/intrinsic_evaluation/our_benchmarks/{core_concept_filename}{embedding_path[dataset]}.pkl', 'rb') as f:
                core_dict = pickle.load(f)

            rev_name_dict = {v:k for k,v in name_dict.items()}
            core_id_dict = {}
            for element in core_dict:
                if element == 'core':
                    core_id_dict[element] = code_dict[rev_name_dict[core_dict[element]]]
                else:
                    codes_list = []
                    for name in core_dict[element]:
                        try:
                            codes_list.append(code_dict[rev_name_dict[name]])
                        except:
                            print(name)
                    core_id_dict[element] = codes_list

            core_concept_id = core_id_dict['core']
            known_pairs = [(core_concept_id, concept) for concept in core_id_dict['complications']
                        ] + [(core_concept_id, concept) for concept in core_id_dict['comorbidities']
                        ] + [(core_concept_id, concept) for concept in core_id_dict['child_diseases']
                        ] 
            known_synonyms = [(core_concept_id, concept) for concept in core_id_dict['synonyms']]
            
            for embedding_name, embedding_tensor in embedding_tensors.items():
                power = evaluate_known_relationships(known_pairs, known_synonyms, embedding_tensor)
                power_results[(core_concept, embedding_name)].append(power)

    # Save results
    with open(f'{WORKING_DIR}/intrinsic_evaluation/results/known_relationship_results_rollup_lvl_{rollup_lvl}{embedding_path[dataset]}.pkl', 'wb') as f:
        pickle.dump(power_results, f)


def main():
    node_similarities(rollup_lvl, dataset)
    print('Tree sim complete', flush = True)
    cooc_similarities(rollup_lvl, dataset)
    print('Cooc sim complete', flush = True)
    comorb_similarities(rollup_lvl, dataset)
    print('Comorb sim complete', flush = True)

if __name__ == "__main__":
    main()