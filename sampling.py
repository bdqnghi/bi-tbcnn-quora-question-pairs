"""Functions to help with sampling trees."""

import pickle
import numpy as np
import random
from tqdm import *
def gen_samples(trees, labels, vectors, vector_lookup):
    """Creates a generator that returns a tree in BFS order with each node
    replaced by its vector embedding, and a child lookup table."""

    # encode labels as one-hot vectors
    label_lookup = {label: _onehot(i, len(labels)) for i, label in enumerate(labels)}
    # print vector_lookup

    for tree in trees:

        nodes = []
        children = []
        label = label_lookup[tree['label']]

        queue = [(tree['tree'], -1)]
        # print queue
        while queue:
            # print "############"
            node, parent_ind = queue.pop(0)
            # print node
            # print parent_ind
            node_ind = len(nodes)
            # print "node ind : " + str(node_ind)
            # add children and the parent index to the queue
            queue.extend([(child, node_ind) for child in node['children']])
            # create a list to store this node's children indices
            children.append([])
            # add this child to its parent's child list
            if parent_ind > -1:
                children[parent_ind].append(node_ind)
            
            n = str(node['node'])
            look_up_vector = vector_lookup[n]
            nodes.append(vectors[int(n)])
        # print "children list length: " + str(len(children))
        yield (nodes, children, label)

def cut_pair_wise(left_inputs,right_inputs):
    random.shuffle(left_inputs)
    random.shuffle(right_inputs)
    range_data = min(len(left_inputs), len(right_inputs))
    return left_inputs[0:range_data], right_inputs[0:range_data]

def generate_zero_pairwise(source,targets):
    source_part = len(source)/len(targets)
    right_data = []
    for target in targets:
        random.shuffle(target)
       
        right_data.extend(target[0:source_part])
       
    left_data, right_data = cut_pair_wise(source, right_data)
    random.shuffle(left_data)
    random.shuffle(right_data)
    return left_data[0:len(left_data)/5],right_data[0:len(right_data)/5]

def patch_data(left_tree, right_tree, vectors, node_map):

    batch_left_nodes, batch_left_children = [], []
    batch_right_nodes, batch_right_children = [], []
    left_nodes = []
    left_children = []
    left_queue = [(left_tree, -1)]
    # print queue
    while left_queue:   
        node, parent_ind = left_queue.pop(0)    
        node_ind = len(left_nodes)   
        # print node
        left_queue.extend([(child, node_ind) for child in node['children']])  
        left_children.append([])    
        if parent_ind > -1:
            left_children[parent_ind].append(node_ind)  
        if node["node"] in node_map:
            node_index = node_map[node["node"]]
        else:
            node_index = node_map["."]
        left_nodes.append(vectors[node_index])

   
    right_nodes = []
    right_children = [] 
    right_queue = [(right_tree, -1)]
      
    while right_queue:   
        node, parent_ind = right_queue.pop(0)    
        node_ind = len(right_nodes)   
        right_queue.extend([(child, node_ind) for child in node['children']])  
        right_children.append([])    
        if parent_ind > -1:
            right_children[parent_ind].append(node_ind)      
        # node_index = int(node['node'])

        if node["node"] in node_map:
            node_index = node_map[node["node"]]
        else:
            node_index = node_map["."]
        right_nodes.append(vectors[node_index])
        
    batch_left_nodes.append(left_nodes)
    batch_left_children.append(left_children)

    batch_right_nodes.append(right_nodes)
    batch_right_children.append(right_children)

    left_nodes, left_children = _pad_batch_siamese(batch_left_nodes, batch_left_children)
    right_nodes, right_children = _pad_batch_siamese(batch_right_nodes, batch_right_children)
    # return _pad_batch_siamese_2_side_for_testing(batch_left_nodes, batch_left_children, batch_right_nodes, batch_right_children)
    return left_nodes, left_children, right_nodes, right_children
    
def _pad_batch_siamese_2_side_for_testing(batch_left_nodes, batch_left_children, batch_right_nodes, batch_right_children):
    return _pad_batch_siamese(batch_left_nodes, batch_left_children), _pad_batch_siamese(batch_right_nodes, batch_right_children)


def _pad_batch_siamese_2_side(batch_left_nodes, batch_left_children, batch_right_nodes, batch_right_children, batch_labels):
    return _pad_batch_siamese(batch_left_nodes, batch_left_children), _pad_batch_siamese(batch_right_nodes, batch_right_children), batch_labels


def _pad_batch_siamese(nodes, children):
    if not nodes:
        return [], [], []
    max_nodes = max([len(x) for x in nodes])
    max_children = max([len(x) for x in children])
    print("Max nodes : " + str(max_nodes))
    print("Max children : " + str(max_children))
    feature_len = len(nodes[0][0])
    child_len = max([len(c) for n in children for c in n])

    nodes = [n + [[0] * feature_len] * (max_nodes - len(n)) for n in nodes]
    # pad batches so that every batch has the same number of nodes
    children = [n + ([[]] * (max_children - len(n))) for n in children]
    # pad every child sample so every node has the same number of children
    children = [[c + [0] * (child_len - len(c)) for c in sample] for sample in children]

    return nodes, children

def batch_random_samples_2_sides(left_trees, right_trees, vectors, node_map, labels, batch_size):
    """Creates a generator that returns a tree in BFS order with each node
    replaced by its vector embedding, and a child lookup table."""

    print("labels length : " + str(len(labels)))
    print("left length : " + str(len(left_trees)))
    print("right length : " + str(len(right_trees)))
    batch_left_nodes, batch_left_children = [], []

    batch_right_nodes, batch_right_children = [], []

    batch_labels = []

    samples = 0

   
    # print node_map
    for i in range(0,len(left_trees)):
      
        
        left_tree = left_trees[i]

        left_nodes = []
        left_children = []
        left_queue = [(left_tree, -1)]
        # print queue
        while left_queue:   
            node, parent_ind = left_queue.pop(0)    
            node_ind = len(left_nodes)   
            # print node
            left_queue.extend([(child, node_ind) for child in node['children']])  
            left_children.append([])    
            if parent_ind > -1:
                left_children[parent_ind].append(node_ind)  
            
            
            # node_index = int(node['node'])
            # print(node["node"])
            # node_index = int(node['node'])
            # print("Fucking node : " + str(node["node"]))
            if node["node"] in node_map:
                node_index = node_map[node["node"]]
            else:
                node_index = node_map["."]
            left_nodes.append(vectors[node_index])
        

        right_tree = right_trees[i]

        right_nodes = []
        right_children = []
        # right_label = right_tree['label']
        right_queue = [(right_tree, -1)]
        # print queue
        while right_queue:   
            node, parent_ind = right_queue.pop(0)    
            node_ind = len(right_nodes)   
            right_queue.extend([(child, node_ind) for child in node['children']])  
            right_children.append([])    
            if parent_ind > -1:
                right_children[parent_ind].append(node_ind)      
            # node_index = int(node['node'])

            if node["node"] in node_map:
                node_index = node_map[node["node"]]
            else:
                node_index = node_map["."]
            right_nodes.append(vectors[node_index])
        
       

        batch_left_nodes.append(left_nodes)
        batch_left_children.append(left_children)
      
        batch_right_nodes.append(right_nodes)
        batch_right_children.append(right_children)
     
        batch_labels.append(labels[i])
        samples += 1

        
        if samples >= batch_size:
            yield _pad_batch_siamese_2_side(batch_left_nodes, batch_left_children, batch_right_nodes, batch_right_children, batch_labels)
            batch_left_nodes, batch_left_children = [], []

            batch_right_nodes, batch_right_children = [], []

            batch_labels = []
            samples = 0

    # if batch_left_nodes and batch_right_labels:
    #     yield _pad_batch_siamese_2_side(batch_left_nodes, batch_left_children, batch_right_nodes, batch_right_children, batch_labels)


def gen_fast_samples(trees, labels, vectors, vector_lookup):
    """Creates a generator that returns a tree in BFS order with each node
    replaced by its vector embedding, and a child lookup table."""

    print("number of trees : "  + str(len(trees)))
    # encode labels as one-hot vectors
    label_lookup = {label: _onehot(i, len(labels)) for i, label in enumerate(labels)}
    # print vector_lookup
    # print vectors
    for tree in trees:

        nodes = []
        children = []
        label_one_hot = label_lookup[tree['label']]
      
        queue = [(tree['tree'], -1)]
        # print queue
        while queue:
            # print "############"
            node, parent_ind = queue.pop(0)
            # print node
            # print parent_ind
            node_ind = len(nodes)
            # print "node ind : " + str(node_ind)
            # add children and the parent index to the queue
            queue.extend([(child, node_ind) for child in node['children']])
            # create a list to store this node's children indices
            children.append([])
            # add this child to its parent's child list
            if parent_ind > -1:
                children[parent_ind].append(node_ind)
            
            node_index = int(node['node'])
            # print "node : " + str(node_index)

            # print vectors[node_index]
            # look_up_vector = vector_lookup[int(node)]
            # print "vector look up : " + str(look_up_vector)
            nodes.append(vectors[node_index])
        # print "children list length: " + str(len(children))
        yield (nodes, children, label_one_hot)

def batch_siamese_samples(gen, batch_size):
    """Batch samples from a generator"""
    nodes, children, labels_one_hot, labels = [], [], [], []
    samples = 0
    for n, c, lo, l in gen:
    
        nodes.append(n)
        children.append(c)
        labels_one_hot.append(lo)
        labels.append(l)
        samples += 1
        if samples >= batch_size:
            yield _pad_batch_siamese(nodes, children,labels_one_hot, labels)
            nodes, children, labels_one_hot, labels = [], [], [], []
            samples = 0

    if nodes:
        yield _pad_batch_siamese(nodes, children,labels_one_hot, labels)



def batch_samples(gen, batch_size):
    """Batch samples from a generator"""
    nodes, children, labels = [], [], []
    samples = 0
    for n, c, l in gen:
        # print n
        # print c
        # print l
        nodes.append(n)
        children.append(c)
        labels.append(l)
        samples += 1
        if samples >= batch_size:
            yield _pad_batch(nodes, children, labels)
            nodes, children, labels = [], [], []
            samples = 0

    if nodes:
        yield _pad_batch(nodes, children, labels)




def _pad_batch(nodes, children, labels):
    if not nodes:
        return [], [], []
    max_nodes = max([len(x) for x in nodes])
    max_children = max([len(x) for x in children])
    feature_len = len(nodes[0][0])
    child_len = max([len(c) for n in children for c in n])

    nodes = [n + [[0] * feature_len] * (max_nodes - len(n)) for n in nodes]
    # pad batches so that every batch has the same number of nodes
    children = [n + ([[]] * (max_children - len(n))) for n in children]
    # pad every child sample so every node has the same number of children
    children = [[c + [0] * (child_len - len(c)) for c in sample] for sample in children]

    return nodes, children, labels

def _onehot(i, total):
    return [1.0 if j == i else 0.0 for j in range(total)]
