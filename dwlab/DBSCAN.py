import numpy as np
import string

def dbscan(D, eps, MinPts):
    labels = [0] * len(D)
    core_points = []
    outliers = []

    C = 0

    for P in range(0, len(D)):
        if not (labels[P] == 0):
            continue

        NeighborPts = region_query(D, P, eps)
        if len(NeighborPts) < MinPts:
            labels[P] = -1
            outliers.append(P)
        else:
            C += 1
            core_points.append(P)
            grow_cluster(D, labels, core_points, P, NeighborPts, C, eps, MinPts)

    return labels, core_points, outliers

def grow_cluster(D, labels, core_points, P, NeighborPts, C, eps, MinPts):
    labels[P] = C
    core_points.append(P)

    i = 0
    while i < len(NeighborPts):
        Pn = NeighborPts[i]

        if labels[Pn] == -1:
            labels[Pn] = C
            core_points.append(Pn)

        elif labels[Pn] == 0:
            labels[Pn] = C
            core_points.append(Pn)

            PnNeighborPts = region_query(D, Pn, eps)

            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts

        i += 1

def region_query(D, P, eps):
    neighbors = []
    for Pn in range(0, len(D)):
        if np.linalg.norm(np.array(D[P]) - np.array(D[Pn])) < eps:
            neighbors.append(Pn)

    return neighbors

# Input points
points = [(2,10), (2,5), (8,4), (5,8), (7,5), (6,4), (1,2), (4,9)]
X = np.array(points)

# Assign alphabet labels to points
alphabet_labels = list(string.ascii_uppercase)
point_labels = {i: label for i, label in enumerate(alphabet_labels)}

# Run DBSCAN
my_labels, core_points, outliers = dbscan(X, eps=2, MinPts=2)

# Remove duplicates
core_points = list(set(core_points))
outliers = list(set(outliers))

# Print clusters
print("Cluster 1 (Core Points):")
for point_index in core_points:
    print(f"Point {point_labels[point_index]}: {points[point_index]}")
print()

print("Cluster 2 (Outliers):")
for point_index in outliers:
    print(f"Point {point_labels[point_index]}: {points[point_index]}")
