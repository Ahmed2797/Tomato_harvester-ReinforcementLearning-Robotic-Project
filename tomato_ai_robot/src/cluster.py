# src/cluster.py
import numpy as np
from sklearn.cluster import DBSCAN

def cluster_tomatoes(tomatoes):
    """
    Robust clustering for robotics pipeline.
    Works with:
    - pos (3D)
    - center (2D fallback)
    """

    if len(tomatoes) == 0:
        return tomatoes

    coords = []

    for t in tomatoes:
        # SAFE extraction (no crash)
        if "pos" in t and t["pos"] is not None:
            coords.append(t["pos"][:2])
        elif "center" in t:
            coords.append(t["center"])
        else:
            coords.append((0, 0))  # fallback safety

    coords = np.array(coords)

    # DBSCAN clustering
    labels = DBSCAN(eps=80, min_samples=2).fit(coords).labels_

    # attach cluster id
    for i, t in enumerate(tomatoes):
        t["cluster_id"] = int(labels[i])

    return tomatoes