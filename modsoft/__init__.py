from ._modsoft import ModSoft
from .modsoft import PythonModSoft
import numpy as np
from scipy import sparse
import networkx as nx


def get_modsoft_object(graph, learning_rate=1., init_part=None,
                       n_communities=None, bias=0., resolution=1.):

    if type(graph) == sparse.csr_matrix:
        adj_matrix = graph
    elif type(graph) == np.ndarray:
        adj_matrix = sparse.csr_matrix(graph)
    elif type(graph) == nx.classes.graph.Graph:
        adj_matrix = nx.adj_matrix(graph)
    else:
        raise TypeError("The argument should be a Numpy Array or a Compressed Sparse Row Matrix.")

    if init_part is None:
        init_part = np.arange(adj_matrix.shape[0], dtype=np.int)

    if n_communities is None:
        n_communities = adj_matrix.shape[0]

    return ModSoft(adj_matrix.shape[0],
                   adj_matrix.indices, adj_matrix.indptr, np.array(adj_matrix.data, dtype=np.float),
                   n_communities, learning_rate, bias, init_part, resolution)


def get_python_modsoft_object(graph, learning_rate=1., init_part=None,
                              n_communities=None, bias=0., resolution=1.):

    if type(graph) == sparse.csr_matrix:
        adj_matrix = graph
    elif type(graph) == np.ndarray:
        adj_matrix = sparse.csr_matrix(graph)
    elif type(graph) == nx.classes.graph.Graph:
        adj_matrix = nx.adj_matrix(graph)
    else:
        raise TypeError("The argument should be a Numpy Array or a Compressed Sparse Row Matrix.")

    if init_part is None:
        init_part = np.arange(adj_matrix.shape[0], dtype=np.int)

    if n_communities is None:
        n_communities = adj_matrix.shape[0]

    return PythonModSoft(adj_matrix.shape[0],
                         adj_matrix.indices, adj_matrix.indptr, np.array(adj_matrix.data, dtype=np.float),
                         n_communities, learning_rate, bias, init_part, resolution)
