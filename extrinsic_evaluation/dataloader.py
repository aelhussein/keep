import torch
ROOT_DIR = '/gpfs/commons/projects/ukbb-gursoylab'
WORKING_DIR = f'{ROOT_DIR}/aelhussein'
import sys
sys.path.append(f'{WORKING_DIR}/extrinsic_evaluation')
from transformer_model import *
from configs import * #Hyperparams
import os
from sklearn.model_selection import train_test_split
import pickle
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

def set_all_seeds(seed=42):
    """Set all seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  


# Dictionaries for the index of the array and the omop concept it references
with open(f'{WORKING_DIR}/datasets/vocab_dict/code2id_ukbb_omop_rollup_lvl_{rollup_lvl}{dataset}.pickle', 'rb') as f:
   code_dict = pickle.load(f)
inv_code_dict = {v: k for k, v in code_dict.items()}
codes = list(code_dict.keys())
id_dict = code_dict
id_dict["<PAD>"]= vocab_size

#Dataset
class SentenceDataset(Dataset):
    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        
    def __len__(self):
        return len(self.X_train)
    
    def __getitem__(self, index):
        sentence = self.X_train[index]
        label = self.Y_train[index]
        return sentence, label

def collate_fn(batch):
    sentences, labels = zip(*batch)
    sentences_tensor = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(s, dtype=torch.long) for s in sentences], 
        batch_first=True, 
        padding_value=id_dict["<PAD>"]
    )
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return sentences_tensor, labels_tensor

def load_data(outcome, val_split, subsample_majority=False, subsample_ratio=5.0, random_state = 42):
    #Get labels and sentences from dictionaries
    dico_cohort = f'{WORKING_DIR}/cohorts_patients/outcome_{outcome}/dictionary_label.pkl'
    with open(dico_cohort, 'rb') as file: 
        dico_cohort = pickle.load(file)
    dico_sentences = f'{WORKING_DIR}/cohorts_patients/outcome_{outcome}/dictionary_sentences.pkl'
    with open(dico_sentences, 'rb') as file:
        dico_sentences = pickle.load(file)
    
    rng = np.random.RandomState(random_state)

    # Create dataset    
    sentences = []
    label_values = []
    for key in sorted(dico_sentences.keys()):
        sorted_codes = sorted([k for k in dico_sentences[key] if k in codes])
        if len(sorted_codes) > 20:
            sorted_codes = np.random.choice(sorted_codes, size=20, replace=False)
        sentences.append([id_dict[k] for k in sorted_codes])
        label_values.append(dico_cohort[key])
    sentences = np.array(sentences, dtype=object)
    label_values = np.array(label_values)
    
    # Subsample majority class if requested
    if subsample_majority:
        # Find indices for each class
        majority_idx = np.where(label_values == 0)[0]
        minority_idx = np.where(label_values == 1)[0]
        
        # Calculate number of samples to keep from majority class
        n_minority = len(minority_idx)
        n_majority_keep = int(n_minority * subsample_ratio)
        
        # Randomly sample from majority class
        majority_idx_subsampled = rng.choice(majority_idx,  size=n_majority_keep, replace=False)
        
        # Combine indices and sort them
        keep_idx = np.concatenate([majority_idx_subsampled, minority_idx])
        keep_idx.sort()
        
        # Update data
        sentences = sentences[keep_idx]
        label_values = label_values[keep_idx]
   
    X_train, X_test, Y_train, Y_test = train_test_split(sentences, label_values, test_size=0.2, random_state=random_state, stratify=label_values)
    
    # Class weights
    class_counts = torch.bincount(torch.tensor(Y_train))
    total_samples = len(Y_train)
    num_classes = len(class_counts)

    weights = total_samples / (num_classes * class_counts)
    weights = weights / weights.sum()

    if val_split:
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2,  random_state=random_state, stratify=Y_train)

        # Create train and val datasets
        dataset = SentenceDataset(X_train, Y_train)
        dataset_val = SentenceDataset(X_val, Y_val)

        # Create data loaders with collate_fn
        dataloader = DataLoader(dataset, batch_size=batch_size,  collate_fn=collate_fn, 
            shuffle=True, generator=torch.Generator().manual_seed(42),
            worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id),)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, collate_fn=collate_fn, 
            shuffle=False, generator=torch.Generator().manual_seed(42),
            worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id),)

        return dataloader, dataloader_val, weights
    
    else:
        train_dataset = SentenceDataset(X_train, Y_train)
        test_dataset = SentenceDataset(X_test, Y_test)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                collate_fn=collate_fn, shuffle=True, generator=torch.Generator().manual_seed(42), 
                                worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                collate_fn=collate_fn, shuffle=False, generator=torch.Generator().manual_seed(42),
                                worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id),)

        return train_loader, test_loader, weights
