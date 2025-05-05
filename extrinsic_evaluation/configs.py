import torch
ROOT_DIR = '/gpfs/commons/projects/ukbb-gursoylab'
WORKING_DIR = f'{ROOT_DIR}/aelhussein'

epochs = 500
eval_interval = 5
batch_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_head = 4
n_layer = 1
dropout = 0.2
num_classes = 2
rollup_lvl = 5
dataset = '_ct_filter'
pooling_type = 'cls'
vocab_size = 5686 # number of diseases
N_RUNS = 100 # Bootstrap

# Embeddings
path_embed = {
    'bio_clin_bert': f'{WORKING_DIR}/pretrained_embeddings/bio_clin_bert_embeddings_rollup_lvl_{rollup_lvl}{dataset}.pkl',
    'clin_bert': f'{WORKING_DIR}/pretrained_embeddings/clin_bert_embeddings_rollup_lvl_{rollup_lvl}{dataset}.pkl',
    'biogpt': f'{WORKING_DIR}/pretrained_embeddings/biogpt_embeddings_rollup_lvl_{rollup_lvl}{dataset}.pkl',
    'bio_clin_bert_hierarchy': f'{WORKING_DIR}/pretrained_embeddings/bio_clin_bert_embeddings_rollup_lvl_{rollup_lvl}_hierarchy{dataset}.pkl',
    'clin_bert_hierarchy': f'{WORKING_DIR}/pretrained_embeddings/clin_bert_embeddings_rollup_lvl_{rollup_lvl}_hierarchy{dataset}.pkl',
    'biogpt_hierarchy': f'{WORKING_DIR}/pretrained_embeddings/biogpt_embeddings_rollup_lvl_{rollup_lvl}_hierarchy{dataset}.pkl',
    'node2vec': f'{WORKING_DIR}/trained_embeddings/our_embeddings/node2vec_embeddings/n2v_model_1_emb_100d_ukbb_omop_rollup_lvl_{rollup_lvl}{dataset}.pickle',
    'glove': f'{WORKING_DIR}/trained_embeddings/our_embeddings/glove_embeddings/glove_model_1_emb_100d_ukbb_omop_rollup_lvl_{rollup_lvl}{dataset}.pickle',
    'glove_reg': f'{WORKING_DIR}/trained_embeddings/our_embeddings/glove_embeddings/glove_model_1_emb_100d_ukbb_omop_rollup_lvl_{rollup_lvl}{dataset}_REG_0.001.pickle',
    'cui2vec': f'{WORKING_DIR}/trained_embeddings/cui2vec/cui2vec_embeddings/cui2vec_emb_100d_ukbb_omop_hierarchy_rollup_lvl_{rollup_lvl}{dataset}.pkl',
    'gat':f'{WORKING_DIR}//trained_embeddings/gat/gat_embeddings/gat_model_1_emb_100d_ukbb_omop_rollup_lvl_{rollup_lvl}{dataset}.pickle'
}

""" path_embed = {
    'bio_clin_bert': f'{WORKING_DIR}/pretrained_embeddings/bio_clin_bert_embeddings_rollup_lvl_{rollup_lvl}{dataset}.pkl',
    'clin_bert': f'{WORKING_DIR}/pretrained_embeddings/clin_bert_embeddings_rollup_lvl_{rollup_lvl}{dataset}.pkl',
    'biogpt': f'{WORKING_DIR}/pretrained_embeddings/biogpt_embeddings_rollup_lvl_{rollup_lvl}{dataset}.pkl',
    'bio_clin_bert_hierarchy': f'{WORKING_DIR}/pretrained_embeddings/bio_clin_bert_embeddings_rollup_lvl_{rollup_lvl}_hierarchy{dataset}.pkl',
    'clin_bert_hierarchy': f'{WORKING_DIR}/pretrained_embeddings/clin_bert_embeddings_rollup_lvl_{rollup_lvl}_hierarchy{dataset}.pkl',
    'biogpt_hierarchy': f'{WORKING_DIR}/pretrained_embeddings/biogpt_embeddings_rollup_lvl_{rollup_lvl}_hierarchy{dataset}.pkl',
    'node2vec': f'{WORKING_DIR}/trained_embeddings/our_embeddings/node2vec_embeddings/n2v_model_1_emb_100d_ukbb_omop_rollup_lvl_{rollup_lvl}{dataset}.pickle',
    'glove': f'{WORKING_DIR}/trained_embeddings/our_embeddings/glove_embeddings/glove_model_1_emb_100d_ukbb_omop_rollup_lvl_{rollup_lvl}{dataset}.pickle',
    'glove_reg': f'{WORKING_DIR}/trained_embeddings/our_embeddings/glove_embeddings/glove_model_1_emb_100d_ukbb_omop_rollup_lvl_{rollup_lvl}{dataset}_REG_0.001.pickle',
    'cui2vec': f'{WORKING_DIR}/trained_embeddings/cui2vec/cui2vec_embeddings/cui2vec_emb_100d_ukbb_omop_hierarchy_rollup_lvl_{rollup_lvl}{dataset}.pkl',
    'bio_clin_bert_100d':f'{WORKING_DIR}/pretrained_embeddings/bio_clin_bert_embeddings_rollup_lvl_{rollup_lvl}{dataset}_dim_100_lr_0.0001.pkl',
    'clin_bert_100d': f'{WORKING_DIR}/pretrained_embeddings/clin_bert_embeddings_rollup_lvl_{rollup_lvl}{dataset}_dim_100_lr_5e-05.pkl',
    'biogpt_100d': f'{WORKING_DIR}/pretrained_embeddings/biogpt_embeddings_rollup_lvl_{rollup_lvl}{dataset}_dim_100_lr_0.0001.pkl',
    'bio_clin_bert_hierarchy_100d': f'{WORKING_DIR}/pretrained_embeddings/bio_clin_bert_embeddings_rollup_lvl_{rollup_lvl}_hierarchy{dataset}_dim_100_lr_0.0001.pkl',
    'clin_bert_hierarchy_100d': f'{WORKING_DIR}/pretrained_embeddings/clin_bert_embeddings_rollup_lvl_{rollup_lvl}_hierarchy{dataset}_dim_100_lr_0.001.pkl',
    'biogpt_hierarchy_100d': f'{WORKING_DIR}/pretrained_embeddings/biogpt_embeddings_rollup_lvl_{rollup_lvl}_hierarchy{dataset}_dim_100_lr_0.0001.pkl',
} """