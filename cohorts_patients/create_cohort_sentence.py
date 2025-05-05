import pandas as pd
import pickle
import numpy as np
from collections import Counter
from datetime import datetime
import tqdm
import argparse
import os

ROOT_DIR = '/gpfs/commons/projects/ukbb-gursoylab/aelhussein'
WORKING_DIR = f'{ROOT_DIR}/cohorts_patients'
UKBB_DATA_DIR = '/gpfs/commons/datasets/controlled/ukbb-gursoylab/aelhussein/phewas/data/'

def create_output_directory(outcome_code):
    """Create output directory if it doesn't exist."""
    output_dir = os.path.join(WORKING_DIR, f'outcome_{outcome_code}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_data(df_path, conditions_path):
    """Load required dataframes."""
    df_pos_neg = pd.read_csv(df_path, low_memory=False)
    conditions_date = pd.read_csv(conditions_path)
    return df_pos_neg, conditions_date

def print_case_statistics(labels_dict):
    """Print statistics about the positive and negative cases."""
    num_positive = sum(value == 1 for value in labels_dict.values())
    num_negative = sum(value == 0 for value in labels_dict.values())
    
    print("\nCase Statistics:", flush=True)
    print(f"Total patients: {len(labels_dict)}", flush=True)
    print(f"Positive cases: {num_positive}", flush=True)
    print(f"Negative cases: {num_negative}", flush=True)

def process_positive_cases(conditions_date, dictionary_dates):
    """Process positive cases and create their disease histories."""
    disease_histories_pos = {}
    data = conditions_date[['eid', 'concept_id', 'condition_start_date']]
    
    print("\nProcessing positive cases...", flush=True)
    for patient in tqdm.tqdm(dictionary_dates.keys()):
        date_limit = datetime.strptime(dictionary_dates[patient], '%Y-%m-%d')
        disease_sequence = []
        patient_data = data[data['eid'] == patient]
        
        # Validate disease presence with 2+ occurrences
        patient_diseases = patient_data['concept_id'].to_list()
        disease_counts = Counter(patient_diseases)
        disease_list = [id for id, count in disease_counts.items() if count >= 2]
        
        for disease in disease_list:
            date_list = patient_data[patient_data['concept_id'] == disease]['condition_start_date'].to_list()
            if date_list:
                date_objects = [datetime.strptime(date, '%d/%m/%Y') for date in date_list]
                oldest_date = min(date_objects)
                date_objects.remove(oldest_date)
                second_oldest_date = min(date_objects)
                if second_oldest_date < date_limit:
                    disease_sequence.append(disease)
                    
        if len(disease_sequence) > 0:
            disease_histories_pos[patient] = disease_sequence
            
    return disease_histories_pos

def process_negative_cases(conditions_date, positive_cases, all_patients):
    """Process negative cases and create their disease histories."""
    negative_cases = [i for i in all_patients if i not in positive_cases]
    disease_histories_neg = {}
    data = conditions_date[['eid', 'concept_id', 'condition_start_date']]
    
    print("\nProcessing negative cases...", flush=True)
    for patient in tqdm.tqdm(negative_cases):
        patient_data = data[data['eid'] == patient]
        patient_diseases = patient_data['concept_id'].to_list()
        disease_counts = Counter(patient_diseases)
        disease_sequence = [id for id, count in disease_counts.items() if count >= 2]
        
        if len(disease_sequence) > 0:
            disease_histories_neg[patient] = disease_sequence
            
    return disease_histories_neg

def save_dictionaries(output_dir, combined_dict, labels_dict):
    """Save the processed dictionaries to files."""
    print("\nSaving dictionaries...", flush=True)
    with open(os.path.join(output_dir, 'dictionary_sentences.pkl'), 'wb') as f:
        pickle.dump(combined_dict, f)
        
    with open(os.path.join(output_dir, 'dictionary_label.pkl'), 'wb') as f:
        pickle.dump(labels_dict, f)

def main():
    parser = argparse.ArgumentParser(description='Process medical data and create disease history dictionaries.')
    parser.add_argument('--outcome', type=int, required=True, help='Outcome code')
    parser.add_argument('--conditions_path', type=str, 
                       default=f'{UKBB_DATA_DIR}/conditions_rollup_lvl_5.csv',
                       help='Path to conditions rollup file')
    args = parser.parse_args()

    try:
        # Setup
        output_dir = create_output_directory(args.outcome)
        print(f"\nProcessing outcome code: {args.outcome}", flush=True)
        
        # Load data
        df_pos_neg_path = os.path.join(WORKING_DIR, f'outcome_{args.outcome}', 'df_pos_neg.csv')
        print("\nLoading data files...", flush=True)
        df_pos_neg, conditions_date = load_data(df_pos_neg_path, args.conditions_path)
        
        # Create dictionaries
        dictionary_dates = df_pos_neg[df_pos_neg['case']==1].set_index('eid').iloc[:,1].to_dict()
        disease_histories_pos = process_positive_cases(conditions_date, dictionary_dates)
        disease_histories_neg = process_negative_cases(conditions_date, 
                                                     set(dictionary_dates.keys()),
                                                     df_pos_neg['eid'].to_list())
        
        # Combine and save
        combined_dict = disease_histories_neg | disease_histories_pos
        labels_dict = df_pos_neg.set_index('eid')['case'].to_dict()
        save_dictionaries(output_dir, combined_dict, labels_dict)
        
        # Print statistics
        print_case_statistics(labels_dict)
        
        print(f"\nProcessed disease histories:", flush=True)
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}", flush=True)
        raise

if __name__ == "__main__":
    main()