import math
import numpy as np
import torch
import pandas as pd

def find_ccs(icd, ccs_df, cat_str):
    mask = ccs_df['icd'].str.startswith(icd)
    children = ccs_df.loc[mask, cat_str]
    return icd, children.value_counts().idxmax(), children.value_counts().max(), len(children.unique())

def get_ccs(icd, ccs_df, cat_str):
    if icd in ccs_df['icd'].values:
        cat = ccs_df.loc[ccs_df['icd']==icd, cat_str].value_counts().index.tolist()[0]
    else:
        cat = find_ccs(icd, ccs_df, cat_str)[1]
    return cat

def get_code(icd, tree):
    """return closest node object corresponding to icd code"""
    icd = icd.replace(".", "")
    if len(icd)==3: # 3d code: find 3d code in tree, return None if not found
        return tree.find(icd)
    elif len(icd)==4: # 4d code: find 4d code then 3d code, return None if not found
        node = tree.find(icd[:3]+"."+icd[3:])
        if node is None:
            return tree.find(icd[:3])
        else: 
            return node
    else: # 5d code: find 5d code then 4d code then 3d code, return None if not found
        node = tree.find(icd[:3]+"."+icd[3:])
        if node is None:
            node=tree.find(icd[:3]+"."+icd[3])
            if node is None:
                node=tree.find(icd[:3])
        return node

def get_code_omop(icd, icd2omop_dict):
    try:
        return icd2omop_dict[icd]
    except KeyError:
        return None

def rollup(node, depth):
    """return ancestor at level @depth"""
    while node.depth>depth:
        node = node.parent
    return node

def tree_distance(node1, node2):
    if node1 is None or node2 is None:
        return 0
    common_ancestors = set(node1.parents) & set(node2.parents)
    dist = node1.depth + node2.depth
    for ancestor in common_ancestors:
        new_dist = node1.depth + node2.depth - 2*ancestor.depth
        dist = min(dist, new_dist)
    return dist

def omop_tree_distance(concept1, concept2, df):
    if concept1==concept2:
        return 0 
    min_seps = (df[(df['ancestor_concept_id'] == concept1) & (df['descendant_concept_id'] == concept2)]['min_levels_of_separation'].values).tolist() + (df[(df['ancestor_concept_id'] == concept2) & (df['descendant_concept_id'] == concept1)]['min_levels_of_separation'].values).tolist()
    if len(min_seps):
        return min(min_seps)
    else :
        return df['max_levels_of_separation'].max()

def resnik_sim(node1, node2):
    if node1 is None or node2 is None or node1.occurence_count==0 or node2.occurence_count==0 :
        return 0
    common_ancestors = set(node1.parents + [node1]) & set(node2.parents + [node2])
    root_count = node1.root.occurence_count
    minimum = root_count
    for ancestor in common_ancestors:
        minimum = min(minimum, ancestor.occurence_count)
    return -math.log(minimum/root_count)

def lin_sim(node1, node2):
    if node1 is None or node2 is None or node1.occurence_count==0 or node2.occurence_count==0 :
        return 0
    common_ancestors = set(node1.parents + [node1]) & set(node2.parents + [node2])
    root_count = node1.root.occurence_count
    minimum = root_count
    for ancestor in common_ancestors:
        minimum = min(minimum, ancestor.occurence_count)
    return 2*math.log(minimum/root_count)/(math.log(node1.occurence_count/root_count)+math.log(node2.occurence_count/root_count))

def sampled_sim_correlation(sim_func, get_code, tree, cats, embedding_tensor, close_k, neg_k, id_dict, inv_id_dict):
    similarity_array_1 = [] # cosine similarities
    similarity_array_2 = [] # resnik similarities
    for icd_i in cats: #3750
        vec1 = embedding_tensor[id_dict[icd_i]]
        similarity = torch.nn.functional.cosine_similarity(vec1, embedding_tensor, dim=1).squeeze(0)
        topk_similarities, topk_indices = similarity.topk(close_k+1, largest=True)
        topk_similarities, topk_indices = topk_similarities[1:], topk_indices[1:]
        similarity_array_1 += topk_similarities.tolist()
        node_i = get_code(icd_i, tree)
        for id_j in topk_indices.tolist():
            icd_j = inv_id_dict[id_j]
            similarity_array_2.append(sim_func(node_i, get_code(icd_j, tree)))
        random_indices = torch.randint(low=0, high=embedding_tensor.size(0)-1, size=(neg_k,))
        random_similarity = torch.nn.functional.cosine_similarity(vec1, embedding_tensor[random_indices], dim=1).squeeze(0)
        similarity_array_1 += random_similarity.tolist()
        for id_j in random_indices.tolist():
            icd_j = inv_id_dict[id_j]
            similarity_array_2.append(sim_func(node_i, get_code(icd_j, tree)))
    return np.corrcoef(similarity_array_1, similarity_array_2)[0, 1]

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
        self.occurence_count+=n
    
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
            if icd in node.icds :
                return node
        return None

def cooccurrence_sim_correlation(cooccurrence_matrix, embedding_tensor, cats, K1, K2, code_dict, inv_code_dict):
    similarity_array_1 = []  # cosine similarities
    similarity_array_2 = []  # cooccurrence values
    
    for icd_i in cats:
        # Get embedding similarities
        vec1 = embedding_tensor[code_dict[icd_i]]
        similarity = torch.nn.functional.cosine_similarity(vec1, embedding_tensor, dim=1).squeeze(0)
        
        # Get top K1 most similar vectors (excluding self)
        topk_similarities, topk_indices = similarity.topk(K1+1, largest=True)
        topk_similarities = topk_similarities[1:]  # Exclude self
        topk_indices = topk_indices[1:]  # Exclude self
        
        # Add top K1 similarities
        similarity_array_1.extend(topk_similarities.tolist())
        
        # Get cooccurrence values for top K1
        for idx in topk_indices.tolist():
            icd_j = inv_code_dict[idx]
            i_idx = code_dict[icd_i]
            j_idx = code_dict[icd_j]
            cooc_value = cooccurrence_matrix[i_idx, j_idx]
            similarity_array_2.append(float(cooc_value))
        
        # Sample K2 random pairs
        random_indices = torch.randint(low=0, high=embedding_tensor.size(0)-1, size=(K2,))
        random_similarity = torch.nn.functional.cosine_similarity(vec1, embedding_tensor[random_indices], dim=1).squeeze(0)
        
        # Add random similarities
        similarity_array_1.extend(random_similarity.tolist())
        
        # Get cooccurrence values for random pairs
        for idx in random_indices.tolist():
            icd_j = inv_code_dict[idx]
            i_idx = code_dict[icd_i]
            j_idx = code_dict[icd_j]
            cooc_value = cooccurrence_matrix[i_idx, j_idx]
            similarity_array_2.append(float(cooc_value))
    
    return np.corrcoef(similarity_array_1, similarity_array_2)[0, 1]

