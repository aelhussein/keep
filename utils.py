import math
import numpy as np
import torch
import pandas as pd
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def calculate_ic(G):
    """
    Calculate Information Content for each node in the graph.
    IC(c) = -log(p(c)) where p(c) is probability of encountering concept c
    """
    # Get all nodes once
    all_nodes = list(G.nodes())
    total_concepts = len(all_nodes)
    
    # Pre-calculate all ancestors for each node
    print("Calculating ancestor sets...")
    ancestor_dict = {node: set(nx.ancestors(G, node)) | {node} for node in tqdm(all_nodes)}
    
    # Calculate subsumer counts more efficiently
    subsumer_counts = defaultdict(int)
    for ancestors in ancestor_dict.values():
        for ancestor in ancestors:
            subsumer_counts[ancestor] += 1
    
    # Vectorize IC calculation
    ic_values = {concept: -math.log(count / total_concepts) 
                for concept, count in subsumer_counts.items()}
    
    return ic_values, ancestor_dict

def calculate_all_similarities(G, concept_pairs, similarity_type='both', batch_size=10000):
    """
    Optimized calculation of similarities for multiple concept pairs.
    
    Args:
        G: NetworkX directed graph
        concept_pairs: List of tuples [(concept1, concept2), ...]
        similarity_type: 'resnik', 'lin', or 'both'
        batch_size: Number of pairs to process at once
    
    Returns:
        Dictionary of results
    """
    print("Calculating IC values and ancestor sets...")
    ic_values, ancestor_dict = calculate_ic(G)
    
    results = {}
    
    # Process in batches to manage memory
    for i in tqdm(range(0, len(concept_pairs), batch_size), desc="Processing pairs"):
        batch_pairs = concept_pairs[i:i + batch_size]
        
        # Process each batch
        for concept1, concept2 in batch_pairs:
            # Get pre-calculated ancestor sets
            ancestors1 = ancestor_dict.get(concept1)
            ancestors2 = ancestor_dict.get(concept2)
            
            if not ancestors1 or not ancestors2:
                continue
                
            # Find common ancestors
            common_ancestors = ancestors1 & ancestors2
            
            if not common_ancestors:
                continue
            
            # Calculate maximum IC of common ancestors
            lcs_ic = max(ic_values[lcs] for lcs in common_ancestors)
            
            if similarity_type in ['resnik', 'both']:
                resnik = lcs_ic
                results[(concept1, concept2, 'resnik')] = resnik
                results[(concept2, concept1, 'resnik')] = resnik
            
            if similarity_type in ['lin', 'both']:
                # Calculate Lin similarity
                denominator = ic_values[concept1] + ic_values[concept2]
                if denominator > 0:
                    lin = 2 * lcs_ic / denominator
                    results[(concept1, concept2, 'lin')] = lin
                    results[(concept2, concept1, 'lin')] = lin
                else:
                    results[(concept1, concept2, 'lin')] = 0.0
                    results[(concept2, concept1, 'lin')] = 0.0
    
    return results

def batch_process_similarities(G, concept_pairs, similarity_type='both', batch_size=10000):
    """
    Process similarities in batches and yield results to save memory.
    
    Args:
        G: NetworkX directed graph
        concept_pairs: List of tuples [(concept1, concept2), ...]
        similarity_type: 'resnik', 'lin', or 'both'
        batch_size: Number of pairs to process at once
    
    Yields:
        Dictionary of results for each batch
    """
    print("Calculating IC values and ancestor sets...")
    ic_values, ancestor_dict = calculate_ic(G)
    
    # Process in batches
    for i in tqdm(range(0, len(concept_pairs), batch_size), desc="Processing pairs"):
        batch_pairs = concept_pairs[i:i + batch_size]
        batch_results = {}
        
        for concept1, concept2 in batch_pairs:
            ancestors1 = ancestor_dict.get(concept1)
            ancestors2 = ancestor_dict.get(concept2)
            
            if not ancestors1 or not ancestors2:
                continue
                
            common_ancestors = ancestors1 & ancestors2
            
            if not common_ancestors:
                continue
            
            lcs_ic = max(ic_values[lcs] for lcs in common_ancestors)
            
            if similarity_type in ['resnik', 'both']:
                resnik = lcs_ic
                batch_results[(concept1, concept2, 'resnik')] = resnik
                batch_results[(concept2, concept1, 'resnik')] = resnik
            
            if similarity_type in ['lin', 'both']:
                denominator = ic_values[concept1] + ic_values[concept2]
                if denominator > 0:
                    lin = 2 * lcs_ic / denominator
                    batch_results[(concept1, concept2, 'lin')] = lin
                    batch_results[(concept2, concept1, 'lin')] = lin
                else:
                    batch_results[(concept1, concept2, 'lin')] = 0.0
                    batch_results[(concept2, concept1, 'lin')] = 0.0
        
        yield batch_results

def semantic_sim_correlation(semantic_similarities, embedding_tensor, cats, K1, K2, code_dict, inv_code_dict, similarity_type='resnik'):
    """
    Calculate correlation between semantic similarities (Resnik/Lin) and embedding cosine similarities.
    
    Args:
        semantic_similarities: Dictionary with (concept1, concept2, measure) keys and similarity values
        embedding_tensor: Tensor of embeddings
        cats: List of ICD codes to analyze
        K1: Number of top similar pairs to include
        K2: Number of random pairs to include
        code_dict: Dictionary mapping ICD codes to indices
        inv_code_dict: Dictionary mapping indices to ICD codes
        similarity_type: 'resnik' or 'lin'
    """
    similarity_array_1 = []  # cosine similarities
    similarity_array_2 = []  # semantic similarities
    embeddings = embedding_tensor.numpy()
    embedding_similarity_matrix = cosine_similarity(embeddings)
    for icd_i in cats:
            # Get pre-computed similarities for this concept
            similarity = embedding_similarity_matrix[code_dict[icd_i]]
            
            # Get top K1 most similar vectors (excluding self)
            topk_indices = np.argpartition(similarity, -(K1+1))[-(K1+1):]
            topk_indices = topk_indices[np.argsort(similarity[topk_indices])][::-1]
            topk_similarities = similarity[topk_indices]
            
            # Remove self similarity
            topk_similarities = topk_similarities[1:]
            topk_indices = topk_indices[1:]
            
            # Add top K1 similarities
            similarity_array_1.extend(topk_similarities.tolist())
            
            # Get semantic similarities for top K1
            for idx in topk_indices:
                icd_j = inv_code_dict[idx]
                sem_sim_key = (icd_i, icd_j, similarity_type)
                sem_sim = semantic_similarities.get(sem_sim_key)
                if sem_sim is None:
                    sem_sim_key = (icd_j, icd_i, similarity_type)
                    sem_sim = semantic_similarities.get(sem_sim_key, 0.0)
                similarity_array_2.append(float(sem_sim))
            
            # Sample K2 random pairs
            random_indices = np.random.randint(0, embeddings.shape[0]-1, size=K2)
            random_similarities = similarity[random_indices]
            
            # Add random similarities
            similarity_array_1.extend(random_similarities.tolist())
            
            # Get semantic similarities for random pairs
            for idx in random_indices:
                icd_j = inv_code_dict[idx]
                sem_sim_key = (icd_i, icd_j, similarity_type)
                sem_sim = semantic_similarities.get(sem_sim_key)
                if sem_sim is None:
                    sem_sim_key = (icd_j, icd_i, similarity_type)
                    sem_sim = semantic_similarities.get(sem_sim_key, 0.0)
                similarity_array_2.append(float(sem_sim))
        
    return np.corrcoef(similarity_array_1, similarity_array_2)[0, 1]

def semantic_sim_correlation(semantic_similarities, embedding_tensor, cats, K1, K2, code_dict, inv_code_dict, similarity_type='resnik'):
    """
    Calculate mean correlation between semantic similarities (Resnik/Lin) and embedding cosine similarities per disease.
    """
    embeddings = embedding_tensor.numpy()
    embedding_similarity_matrix = cosine_similarity(embeddings)
    
    # Store correlations for each disease
    disease_correlations = {}
    
    for icd_i in cats:
        similarity_array_1 = []  # cosine similarities
        similarity_array_2 = []  # semantic similarities
        
        # Get pre-computed similarities for this concept
        similarity = embedding_similarity_matrix[code_dict[icd_i]]
        
        # Get top K1 most similar vectors (excluding self)
        topk_indices = np.argpartition(similarity, -(K1+1))[-(K1+1):]
        topk_indices = topk_indices[np.argsort(similarity[topk_indices])][::-1]
        topk_similarities = similarity[topk_indices]
        
        # Remove self similarity
        topk_similarities = topk_similarities[1:]
        topk_indices = topk_indices[1:]
        
        # Add top K1 similarities
        similarity_array_1.extend(topk_similarities.tolist())
        
        # Get semantic similarities for top K1
        for idx in topk_indices:
            icd_j = inv_code_dict[idx]
            sem_sim_key = (icd_i, icd_j, similarity_type)
            sem_sim = semantic_similarities.get(sem_sim_key)
            if sem_sim is None:
                sem_sim_key = (icd_j, icd_i, similarity_type)
                sem_sim = semantic_similarities.get(sem_sim_key, 0.0)
            similarity_array_2.append(float(sem_sim))
        
        # Sample K2 random pairs
        random_indices = np.random.randint(0, embeddings.shape[0]-1, size=K2)
        random_similarities = similarity[random_indices]
        
        # Add random similarities
        similarity_array_1.extend(random_similarities.tolist())
        
        # Get semantic similarities for random pairs
        for idx in random_indices:
            icd_j = inv_code_dict[idx]
            sem_sim_key = (icd_i, icd_j, similarity_type)
            sem_sim = semantic_similarities.get(sem_sim_key)
            if sem_sim is None:
                sem_sim_key = (icd_j, icd_i, similarity_type)
                sem_sim = semantic_similarities.get(sem_sim_key, 0.0)
            similarity_array_2.append(float(sem_sim))
        
        # Calculate correlation for this disease
        if sum(similarity_array_2) == 0:
            continue
        elif sum(similarity_array_1) == 0:
            disease_correlations[icd_i] = 0 # If embedding has no similarity then correlation is 0
        else:
            if len(similarity_array_1) > 1:  # Ensure we have enough pairs for correlation
                correlation = np.corrcoef(similarity_array_1, similarity_array_2)[0, 1]
                if not np.isnan(correlation):  # Only store valid correlations
                    disease_correlations[icd_i] = correlation
                else:
                    print(f"Warning: NaN correlation for disease {icd_i}")
                    print("Similarity array 1:", similarity_array_1)
                    print("Similarity array 2:", similarity_array_2)
    
    # Calculate summary statistics
    correlations = np.array(list(disease_correlations.values()))
    mean_correlation = np.mean(correlations)
    
    return mean_correlation

def cooccurrence_sim_correlation(cooccurrence_matrix, embedding_tensor, cats, K1, K2, code_dict, inv_code_dict):
    similarity_array_1 = []  # cosine similarities
    similarity_array_2 = []  # cooccurrence values
    embeddings = embedding_tensor.numpy()
    embedding_similarity_matrix = cosine_similarity(embeddings)
    for icd_i in cats:
        # Get pre-computed similarities for this concept
        similarity = embedding_similarity_matrix[code_dict[icd_i]]
        
        # Get top K1 most similar vectors (excluding self)
        topk_indices = np.argpartition(similarity, -(K1+1))[-(K1+1):]
        topk_indices = topk_indices[np.argsort(similarity[topk_indices])][::-1]
        topk_similarities = similarity[topk_indices]
        
        # Remove self similarity
        topk_similarities = topk_similarities[1:] 
        topk_indices = topk_indices[1:]
        
        # Add top K1 similarities
        similarity_array_1.extend(topk_similarities.tolist())
        
        # Get cooccurrence values for top K1
        i_idx = code_dict[icd_i]
        j_indices = [code_dict[inv_code_dict[idx]] for idx in topk_indices]
        cooc_values = cooccurrence_matrix[i_idx, j_indices]
        similarity_array_2.extend(cooc_values.tolist())
        
        # Sample K2 random pairs
        random_indices = np.random.randint(0, embeddings.shape[0]-1, size=K2)
        random_similarities = similarity[random_indices]
        
        # Add random similarities
        similarity_array_1.extend(random_similarities.tolist())
        
        # Get cooccurrence values for random pairs
        j_indices = [code_dict[inv_code_dict[idx]] for idx in random_indices]
        cooc_values = cooccurrence_matrix[i_idx, j_indices]
        similarity_array_2.extend(cooc_values.tolist())
    
    return np.corrcoef(similarity_array_1, similarity_array_2)[0, 1]


def cooccurrence_sim_correlation(cooccurrence_matrix, embedding_tensor, cats, K1, K2, code_dict, inv_code_dict):
    """
    Calculate correlation between cooccurrence values and embedding cosine similarities per disease.
    """
    embeddings = embedding_tensor.numpy()
    embedding_similarity_matrix = cosine_similarity(embeddings)
    
    # Store correlations for each disease
    disease_correlations = {}
    
    for icd_i in cats:
        similarity_array_1 = []  # cosine similarities
        similarity_array_2 = []  # cooccurrence values
        
        # Get pre-computed similarities for this concept
        similarity = embedding_similarity_matrix[code_dict[icd_i]]
        
        # Get top K1 most similar vectors (excluding self)
        topk_indices = np.argpartition(similarity, -(K1+1))[-(K1+1):]
        topk_indices = topk_indices[np.argsort(similarity[topk_indices])][::-1]
        topk_similarities = similarity[topk_indices]
        
        # Remove self similarity
        topk_similarities = topk_similarities[1:]
        topk_indices = topk_indices[1:]
        
        # Add top K1 similarities
        similarity_array_1.extend(topk_similarities.tolist())
        
        # Get cooccurrence values for top K1
        i_idx = code_dict[icd_i]
        j_indices = [code_dict[inv_code_dict[idx]] for idx in topk_indices]
        cooc_values = cooccurrence_matrix[i_idx, j_indices]
        similarity_array_2.extend(cooc_values.tolist())
        
        # Sample K2 random pairs
        random_indices = np.random.randint(0, embeddings.shape[0]-1, size=K2)
        random_similarities = similarity[random_indices]
        
        # Add random similarities
        similarity_array_1.extend(random_similarities.tolist())
        
        # Get cooccurrence values for random pairs
        j_indices = [code_dict[inv_code_dict[idx]] for idx in random_indices]
        cooc_values = cooccurrence_matrix[i_idx, j_indices]
        similarity_array_2.extend(cooc_values.tolist())
        #Disease has no coocurrences
        if sum(similarity_array_2) == 0:
            continue
        elif sum(similarity_array_1) == 0:
            disease_correlations[icd_i] = 0 # If embedding has no similarity then correlation is 0
        else:
            # Calculate correlation for this disease
            if len(similarity_array_1) > 1:  # Ensure we have enough pairs for correlation
                correlation = np.corrcoef(similarity_array_1, similarity_array_2)[0, 1]
                if not np.isnan(correlation):  # Only store valid correlations
                    disease_correlations[icd_i] = correlation
                else:
                    print(f"Warning: NaN correlation for disease {icd_i}")
                    print("Similarity array 1:", similarity_array_1)
                    print("Similarity array 2:", similarity_array_2)
    
    # Calculate summary statistics
    correlations = np.array(list(disease_correlations.values()))
    mean_correlation = np.mean(correlations)
    
    return mean_correlation


def cosine_similarity_torch(vec1, vec2):
    return torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()

def compute_bootstrap_null_distribution(embedding_tensor, known_pairs, known_synonyms, n_samples=10000):
    """Create a bootstrap distribution of cosine similarities for random pairs across all concepts, excluding known pairs and synonyms."""
    num_concepts = embedding_tensor.size(0)
    distribution = []
    
    # Convert known pairs and synonyms to sets for fast lookup
    excluded_pairs = set(known_pairs + known_synonyms)

    for _ in range(n_samples):
        while True:
            # Randomly sample two indices for concept pairs
            i, j = np.random.choice(num_concepts, size=2, replace=False)
            if (i, j) not in excluded_pairs and (j, i) not in excluded_pairs:
                break
        
        vec1 = embedding_tensor[i]
        vec2 = embedding_tensor[j]
        similarity = cosine_similarity_torch(vec1, vec2)
        distribution.append(similarity)
    
    return np.array(distribution)

def evaluate_known_relationships(known_pairs, known_synonyms, embedding_tensor, alpha=0.05, n_samples=10000):
    """
    For each known relationship, compute cosine similarity and evaluate significance.
    """
    # Generate null distribution excluding known pairs and synonyms
    null_distribution = compute_bootstrap_null_distribution(embedding_tensor, known_pairs, known_synonyms, n_samples=n_samples)
    threshold = np.percentile(null_distribution, 95)
    
    # Evaluate each known relationship in known_pairs
    significant_count = 0
    for (concept1, concept2) in known_pairs:
        vec1 = embedding_tensor[concept1]
        vec2 = embedding_tensor[concept2]
        observed_similarity = cosine_similarity_torch(vec1, vec2)
        
        # Check if the observed similarity is significant
        if observed_similarity > threshold:
            significant_count += 1
    power = significant_count / len(known_pairs)
    return power


class Node(object):
    def __init__(self, concept_id, icds, root) -> None:
        self.concept_id = concept_id
        self.icds = icds
        self.occurence_count = 0
        self.root = root
        if icds == ['root']:
            self.parents = []
        else:
            self.parents = [root]
    
    def add_ancestor(self, ancestor):
        self.parents.append(ancestor)
    
    def add_count(self, n):
        self.occurence_count += n
    
    def add_occurence(self, n):
        self.add_count(n)
        for ancestor in self.parents:
            ancestor.add_count(n)


class Graph(Node):
    def __init__(self) -> None:
        super().__init__(0, ['root'], self)
        self.nodes = []

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.append(node)
    
    def get_node(self, concept_id):
        codes = [node.concept_id for node in self.nodes]
        try:
            return self.nodes[codes.index(concept_id)]
        except ValueError:
            return None
    
    def find(self, icd):
        for node in self.nodes:
            if icd in node.icds:
                return node
        return None
    
    def number_of_nodes(self):
        """
        Returns the number of nodes in the graph.
        """
        return len(self.nodes)
    
    def number_of_edges(self):
        """
        Returns the number of edges in the graph.
        Each edge is defined as a parent-child relationship.
        """
        edge_count = 0
        for node in self.nodes:
            edge_count += len(node.parents)
        return edge_count
