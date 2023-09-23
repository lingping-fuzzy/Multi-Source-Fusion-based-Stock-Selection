'''
@author: Brecht Laperre
@description: dtw-measure algorithm
'''
import sys
import numpy as np
from dtw_base import *

def _compute_cost(D):

    D[0, :] = np.cumsum(D[0, :])
    D[:, 0] = np.cumsum(D[:, 0])

    for i in range(1, D.shape[0]):
        new_min = np.min(D[i-1:i+1, 0])
        for j in range(1, D.shape[1]):
            if D[i-1, j] < new_min:
                new_min = D[i-1, j]
                D[i, j] = D[i, j] + new_min
                continue
            else:
                D[i, j] = D[i, j] + new_min
                new_min = D[i-1, j] if D[i-1, j] < D[i, j] else D[i, j]

    return D

def _find_path(D):
    assert len(D[:, 0])
    assert len(D[0, :])

    path = np.zeros((2, D.shape[0] + D.shape[1]), dtype=int)

    loc_i, loc_j = D.shape[0]-1, D.shape[1]-1
    ind = D.shape[0] + D.shape[1] - 1
    path[:, ind] = np.array([loc_i, loc_j])
    while (loc_i != 0) and (loc_j != 0):
        v = np.array([D[loc_i-1, loc_j-1], D[loc_i-1, loc_j], D[loc_i, loc_j-1]])
        _min = np.argmin(v)
        if _min == 0 or _min == 1:
            loc_i -= 1
        if _min == 0 or _min == 2:
            loc_j -= 1
        ind -= 1
        path[:, ind] = np.array([loc_i, loc_j])

    if loc_j != 0:
        for k in range(loc_j-1, 0, -1):
            ind -= 1
            path[:, ind] = np.array([loc_i, k])
        ind -= 1
    if loc_i != 0:
        for k in range(loc_i-1, 0, -1):
            ind -= 1
            path[:, ind] = np.array([k, loc_j])
        ind -= 1

    return D[-1, -1], path[:, ind:]


def _compute_dist(T, P):
    D = np.zeros((len(T), len(P)))

    for i, t in enumerate(T):
        D[i, i:] = np.sqrt((P[i:] - t) ** 2)

    if len(T) > len(P):
        D[len(P):, -1] = np.sqrt((T[len(P):] - P[-1]) ** 2)

    return D


def _compute_dist_windowed(T, P, window):
    assert window < len(P)
    assert len(P) == len(T)
    D = np.ones((len(T), len(P)))*np.infty
    D_long =  np.ones(( len(P), window+1))*np.infty
    for i, t in enumerate(T):
        #if i <= window:
        #    D[i, :i+1] = np.sqrt((P[:i+1] - t) ** 2)
        if len(P) - i <= window:
            D[i, i:] = np.sqrt((P[i:] - t) ** 2)
        else:
            D[i, i:i+window+1] = np.sqrt((P[i:i+window+1] - t) ** 2)
            D_long[i, :] = np.sqrt((P[i:i+window+1] - t) ** 2)

    if len(T) > len(P):
        D[len(P):, -1] = np.sqrt((T[len(P):] - P[-1]) ** 2)

    return D.T


def dtw_measure(T, P, w=None, asymmetric=False):
    '''Modified DTW, with constraint that prevents mapping predictions to truth values in the future
    Input:
        T: True timeseries
        P: Prediction
        w: windowsize
    Output:
        D: matrix of size T x P containing the cost of the mapping
        path: least cost path mapping P to T
        cost: DTW cost of the path
    '''
    assert len(T)
    assert len(P)

    if len(T) > 1000 or len(P) > 1000:
        D, path, cost = eff_dtw_measure(T, P, w, asymmetric)
        return path, cost


    if w is None:
        D = _compute_dist(P, T)
    else:
        D = _compute_dist_windowed(P, T, w)
    D = _compute_cost(D)

    cost, path = _find_path(D)

    return D, path, cost


def main_test(sdata, preds, win=2):
    if len(sdata) <= 1000:
        DD, PATH, COST = dtw_measure(preds, sdata, w=win)
    else:
        PATH, COST = dtw_measure(preds, sdata, w=win)
        return PATH, COST
    print(DD)
    print('Path')
    print(PATH)
    print('cost .')
    print(COST)
    return DD, PATH, COST


def test_timechange(PATH1):
    bins, counts = np.unique(abs(PATH1[0, :] - PATH1[1, :]), return_counts=True)
    print('time slot slow than the Real data ')
    print(PATH1[0, :] - PATH1[1, :])
    print('occured time slow in \'secd\'')
    print(bins)  # differ windows left and right , windows 0, or 1,
    print('occured frequency of time slow')
    print(counts)  # how many values in that categories no differ, with 1 value differ

def main_test_run(): # LSTM_rank1.ippynb
    QQ = np.array([11, 17, 13, 8, 10, 5, 4, 5, 6, 9, 6, 7, 5, 4, 3])
    SS = np.array([12, 15, 18, 12, 7, 5, 2, 3, 1, 8, 5, 9, 4, 1, 4])
    DD, PATH, COST = main_test(QQ, SS, win=3)
    test_timechange(PATH)
    # draw_path(PATH, QQ, SS)

if __name__ == '__main__':
    main_test_run()