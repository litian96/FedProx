import numpy as np

import tensorflow as tf

def __num_elems(shape):
    '''Returns the number of elements in the given shape

    Args:
        shape: TensorShape
    
    Return:
        tot_elems: int
    '''
    tot_elems = 1
    for s in shape:
        tot_elems *= int(s)
    return tot_elems

def graph_size(graph):
    '''Returns the size of the given graph in bytes

    The size of the graph is calculated by summing up the sizes of each
    trainable variable. The sizes of variables are calculated by multiplying
    the number of bytes in their dtype with their number of elements, captured
    in their shape attribute

    Args:
        graph: TF graph
    Return:
        integer representing size of graph (in bytes)
    '''
    tot_size = 0
    with graph.as_default():
        vs = tf.trainable_variables()
        for v in vs:
            tot_elems = __num_elems(v.shape)
            dtype_size = int(v.dtype.size)
            var_size = tot_elems * dtype_size
            tot_size += var_size
    return tot_size

def process_sparse_grad(grads):
    '''
    Args:
        grads: grad returned by LSTM model (only for the shakespaere dataset)
    Return:
        a flattened grad in numpy (1-D array)
    '''

    indices = grads[0].indices
    values =  grads[0].values
    first_layer_dense = np.zeros((80,8))
    for i in range(indices.shape[0]):
        first_layer_dense[indices[i], :] = values[i, :]

    client_grads = first_layer_dense
    for i in range(1, len(grads)):
        client_grads = np.append(client_grads, grads[i]) # output a flattened array


    return client_grads

def process_grad(grads):
    '''
    Args:
        grads: grad 
    Return:
        a flattened grad in numpy (1-D array)
    '''

    client_grads = grads[0]

    for i in range(1, len(grads)):
        client_grads = np.append(client_grads, grads[i]) # output a flattened array


    return client_grads

def cosine_sim(a, b):
    '''Returns the cosine similarity between two arrays a and b
    '''  
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product * 1.0 / (norm_a * norm_b)  




