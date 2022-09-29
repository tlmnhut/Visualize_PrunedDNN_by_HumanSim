# ANNA'S CODE

import numpy as np
from scipy.io import loadmat
import argparse

from utils import upper_tri


def cor_order(mat):
    """
      Returns the upper triangle elements of the similarity matrix computed
      from DNN embeddings, removing one feature at a time

      Parameters
      ----------
      mat : numpy matrix
            N by M feature embedding matrix with N observations (images) in
            M dimensions (based on the network from which embeddings are extracted).

      Returns
      ---------

      ret : numpy matrix
          M x (N*(N-1)/2) dimensions
          Returns the upper triangle elements of the similarity matrix computed
          from DNN embeddings, removing one feature at a time (every SM is computed
          with only one feature removed)

    """
    n_cols = int(mat.shape[0] * (mat.shape[0] - 1) / 2)
    n_rows = mat.shape[1]
    ret = np.empty(shape=(n_rows, n_cols))

    for k in range(mat.shape[1]):
        d_drop = np.delete(mat, k, axis=1)
        m1 = np.corrcoef(d_drop)
        ret[k, :] = upper_tri(m1)

    return ret


def compute_2oi(dnn_mat, human_rsm, sorted_features):
    '''
        Input:
        dnn_mat: N x M matrix containing N image representations each consisting of M features.
        human_mat: N x N matrix containing human similarity judgements.
        sorted_features: list of features sorted from the least to the most important

        Output:
        List containing the 2OI values obtained after inserting one feature after
        the other, starting from the MOST important one.
    '''

    features = [feature for feature in reversed(sorted_features)]
    r2s = []

    for i in range(len(sorted_features)):
        dnn_rdm = upper_tri(np.nan_to_num(np.corrcoef(dnn_mat[:, features[:i]])))
        r2 = np.nan_to_num(np.corrcoef(human_rsm, dnn_rdm))[0][1] ** 2
        r2s.append(r2)

    return r2s


# PRIYA'S METHOD
def compute_feature_ranking(dnn_mat, fmri_rsm):
    """
    Rank the importance of each feature from least to most
    :param dnn_mat: numpy 2D matrix, activations from DNN
    :param fmri_rsm: numpy 1D array, RDM of human fMRI data
    :return: list of indices of features, sorted from least to most important
    """

    rdms_vec = cor_order(dnn_mat)
    initial_2oi = np.corrcoef(upper_tri(np.corrcoef(dnn_mat)), fmri_rsm)[0][1]**2
    n_features = dnn_mat.shape[1]
    diff_vec = np.zeros((rdms_vec.shape[0]))

    for i in range(n_features):
        new_2oi = np.corrcoef(rdms_vec[i], fmri_rsm)[0][1]**2
        diff = initial_2oi-new_2oi # a high value means high importance of the feature
        diff_vec[i] = diff
  
    return list(np.argsort(diff_vec))


def prune_network(dnn_mat, fmri_rsm):
    """
       Input:
       dnn_mat: N x M matrix containing the initial dnn representations with all features.
       human_mat: N x N matrix with human similarity judgements.

       Returns:
       train_2oi: folds x M array with 2oi computed on the training set after every feature insertion
       test_2oi: folds x M array with 2oi computed on the test set after every feature insertion
       train_max2OI: folds-dimension array with the maximum 2OI obtained for each fold on the training set
       test_max2OI: folds-dimension array with the maximum 2OI obtained for each fold on the test set
       n_ret_features: folds-dimension array with the optimal number of features for every fold
    """

    sorted_features = compute_feature_ranking(dnn_mat, fmri_rsm)

    # 2OI on training data
    r2s = compute_2oi(dnn_mat, fmri_rsm, sorted_features)

    n_ret_ft = np.argmax(r2s)

    max2OI = r2s[n_ret_ft]
    print('Final 2OI:', max2OI)

    ret_features = sorted_features[-n_ret_ft:]

    return ret_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", help="either 1 or 2")
    parser.add_argument("--brain_area", help="either PPA or FFA or vTC")
    args = parser.parse_args()

    brain_area = args.brain_area
    dataset_name = 'set' + args.set

    # load unpruned activations from full set
    dnn = np.load('./data/set'+args.set+'_repr_classifier_3.npy')

    # load fMRI data, then average for all participants
    fmri_mat = loadmat('./data/FMRI.mat', simplify_cells=True)['FMRI']
    select_roi = fmri_mat[brain_area][dataset_name]
    sim_scores = select_roi['pairwisedistances']
    fmri_data = np.mean(sim_scores, axis=0)

    # prune
    features = prune_network(dnn, fmri_data)
    np.save('./data/set'+args.set+'_'+brain_area+'_retained_feat', features)
    print('Retained features are saved in ./data')
  
