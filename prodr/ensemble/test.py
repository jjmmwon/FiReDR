import numpy as np
import scipy.sparse as sp

# -----------------------------
# function under test
# -----------------------------
from scipy.sparse.csgraph import connected_components


def split_micro_clusters(
    collision_matrix: sp.csr_matrix, threshold: int
) -> tuple[list[np.ndarray], list[sp.csr_matrix]]:

    adjacency_matrix = collision_matrix >= threshold

    n_components, labels = connected_components(
        adjacency_matrix, directed=False, return_labels=True
    )

    label_to_indices: dict[int, list[int]] = {
        label: [] for label in range(n_components)
    }
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    micro_clusters_instances = [
        np.array(indices) for indices in label_to_indices.values()
    ]
    micro_clusters = [
        collision_matrix[indices][:, indices] for indices in micro_clusters_instances
    ]

    return micro_clusters_instances, micro_clusters


# -----------------------------
# test case
# -----------------------------
def test_split_micro_clusters():
    """
    Collision matrix example:

    Nodes: 0 1 2 3 4 5

    - {0,1,2} have collision = 3
    - {3,4} have collision = 2
    - 5 is isolated
    """
    C = np.array(
        [
            [0, 3, 3, 0, 0, 0],
            [3, 0, 3, 0, 0, 0],
            [3, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 0],
            [0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    collision_matrix = sp.csr_matrix(C)

    # ---- threshold = 3 ----
    instances, submatrices = split_micro_clusters(collision_matrix, threshold=3)

    print("threshold = 3")
    for inst, sub in zip(instances, submatrices):
        print("cluster indices:", inst.tolist())
        print(sub.toarray())
        print()

    # ---- threshold = 2 ----
    instances, submatrices = split_micro_clusters(collision_matrix, threshold=2)

    print("threshold = 2")
    for inst, sub in zip(instances, submatrices):
        print("cluster indices:", inst.tolist())
        print(sub.toarray())
        print()


if __name__ == "__main__":
    test_split_micro_clusters()
