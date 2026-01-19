from collections import Counter

import numpy as np
import scipy.sparse as sp

from prodr.ensemble.components import Node


def generate_cooccurr_mtx(
    cooccurr_list: list[list[int]], n_samples: int
) -> sp.csr_array:
    """
    Generate cooccurrence count matrix from a single tree.
    Args:
        cooccurr_list: List of cooccurr_cnt lists from a single tree.
            indexed by [node_idx][data_point_indices].
            cooccurr_list[node_idx] means the list of data point indices
            that fall into the same node.
        n_samples: Total number of samples.
    Returns:
        Cooccurrence matrix of shape (n_samples, n_samples).
    """
    row_chunks = []
    col_chunks = []

    for cooccurr_indices in cooccurr_list:
        indices = np.asarray(cooccurr_indices, dtype=np.int64)
        m = indices.size
        if m <= 1:
            continue

        rr, cc = np.triu_indices(m, k=1)
        row_chunks.append(indices[rr])
        col_chunks.append(indices[cc])

    if not row_chunks:
        return sp.csr_array((n_samples, n_samples), dtype=np.int32)

    rows = np.concatenate(row_chunks)
    cols = np.concatenate(col_chunks)
    data = np.ones(rows.size, dtype=np.int32)
    C = sp.coo_matrix(
        (data, (rows, cols)), shape=(n_samples, n_samples), dtype=np.int32
    ).tocsr()
    C.sum_duplicates()
    C.eliminate_zeros()

    return C + C.T


def generate_cooccurr_acc_mtx(
    cooccurr_cnt_list: list[list[list[int]]], n_samples: int
) -> sp.csr_array:
    """
    Generate accumulated cooccurrence count matrix from multiple trees.
    Args:
        cooccurr_cnt_list: List of cooccurr_cnt lists from multiple trees.
            indexed by [tree_idx][node_idx][data_point_indices].
        n_samples: Total number of samples.
    """
    C_total = sp.csr_array((n_samples, n_samples), dtype=np.int32)

    for cooccurr_list in cooccurr_cnt_list:
        C_tree = generate_cooccurr_mtx(cooccurr_list, n_samples)
        C_total += C_tree

    return C_total


def count_cooccurrence(Nodes: list[Node], threshold: int) -> dict[int, int]:
    counter = Counter()
    for node in Nodes:
        counter.update(node.indices)

    return {idx: count for idx, count in counter.items() if count >= threshold}
