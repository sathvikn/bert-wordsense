import numpy as np
import torch

def euc_dist(v1, v2):
    if type(v1) == torch.Tensor:
        v1 = v1.numpy()
    if type(v2) == torch.Tensor:
        v2 = v2.numpy()
    return np.sqrt(np.sum((v1 - v2)**2))

def find_closest_distance(e1_lst, e2_lst, fn):
    #fn can be either np.mean or minimum
    return fn([min([euc_dist(e1, e2) for e2 in e2_lst]) for e1 in e1_lst])

def centroid(arr):
    arr = lst_to_np(arr)
    length, dim = arr.shape
    return np.array([np.sum(arr[:, i])/length for i in range(dim)])

def lst_to_np(arr):
    return np.array([t.numpy() for t in arr])

def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def cs_centroids(s1, s2):
    return cosine_sim(centroid(s1), centroid(s2))

def dist_centroids(s1, s2):
    return euc_dist(centroid(s1), centroid(s2))
