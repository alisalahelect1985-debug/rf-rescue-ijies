import numpy as np
from core import csr_from_edges


class G:
    """Small explicit bipartite graph with the same interface as Geometry."""
    def __init__(self, K, T, edges):
        self.K, self.T = K, T
        self.edges = np.array(edges, int).reshape(-1, 2)
        self.tptr, self.tflw, self.fptr, self.ftok = csr_from_edges(K, T, self.edges)
        self.deg = np.diff(self.tptr)
