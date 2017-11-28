# Omid55
# My utilities

# Imports
import numpy as np
import pyemd
import heapq



def compute_emd(P, Q):
    k = len(P)
    v1 = np.ones(k) / k
    v2 = np.zeros(k)
    D = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            D[i][j] = abs(P[i] - Q[j])
    return pyemd.emd(v1, v2, D)


def find_k_largest(arr, k):
    if len(arr) < k:
        raise ValueError('The length of arr is smaller than k, len(arr):%d, k:%d' % (len(arr), k))
    h = list(arr.copy())
    h = [-i for i in h]
    heapq.heapify(h)
    result = np.zeros(k)
    for i in range(k):
        result[i] = -heapq.heappop(h)
    return result