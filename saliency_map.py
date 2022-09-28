"""Saliency map
Usage:
  saliency_map.py --brain_area=<str> --filter_size=<int> --use_pruning=<int>
  saliency_map.py (-h | --help)
  saliency_map.py --version

Options:
  -h --help                    Show this screen.
  --version                    Show version.
  --brain_area=<str>           Name of brain area, either PPA, or FFA, or None.
  --filter_size=<int>          Filter size.
  --use_pruning=<int>          Use pruned or unpruned model
"""

from docopt import docopt
import pandas as pd
import numpy as np
import copy
import pathlib
import glob
from tqdm import tqdm
from scipy.io import loadmat
from scipy.stats import pearsonr
from sklearn import preprocessing
from multiprocessing import cpu_count, Pool
import torchvision.models as models

from utils import extract_masked_repr, upper_tri, show_heatmap


def get_human_sim_arr(brain_area, dataset_name, chosen_img_idx_list=[], restore_sim_mat=False):
    """
    Average the human (fMRI) RDM of many participants. RDM is selected from a specific brain area such as FFA, PPA.
    :param brain_area: str, e.g. FFA, PPA, vTC
    :param dataset_name: str, either set1 or set2
    :param chosen_img_idx_list: list, index of images chosen to form a smaller Representational Dissimilarity Matrix (RDM)
    Only use in case that you want to select a subset of images and to construct a small RDM
    :param restore_sim_mat: bool, whether to restore the flattened 1D RDM to the full 2D RDM
    :return: np.ndarray, averaged human RDM from a selective brain area
    """

    # load fMRI data, then average the data from multiple participants
    fmri_data = loadmat('./data/FMRI.mat', simplify_cells=True)['FMRI']
    select_roi = fmri_data[brain_area][dataset_name]
    dissim_scores = select_roi['pairwisedistances']
    avg_dissim_scores = np.mean(dissim_scores, axis=0)

    # restore the 2D RDM. First, fill dissimilarity values in the upper triangle part.
    # Second, copy the upper triangle part to the lower triangle part.
    chunk_list = []
    start_idx = 0
    avg_dissim_scores = list(avg_dissim_scores)
    for i in range(143, -1, -1):
        chunk = [0] * (144 - i) + avg_dissim_scores[start_idx:start_idx + i]
        chunk_list.append(chunk)
        start_idx += i
    avg_rdm = np.array(chunk_list)
    avg_rdm += avg_rdm.T

    if chosen_img_idx_list:
        avg_rdm = avg_rdm[:, chosen_img_idx_list][chosen_img_idx_list, :]

    if not restore_sim_mat:
        return upper_tri(avg_rdm)

    return avg_rdm


def compute_corr_masked(repr_mat, repr_full_set, keep_dim, img_idx):
    """
    Compute the Pearson correlation between each instance in the collection of activations from an image
    and all other unmasked images. An instance is an activation vector from unmasked or masked image.
    :param repr_mat: numpy 2D array, activations of the unmasked and masked images
    :param repr_full_set: numpy 2D array, activations of all original images from the full dataset
    :param keep_dim: list, list of indices of features that we want to keep. Only apply in the case of pruned network.
    :param img_idx: int, index of the image
    :return: numpy 2D array, shape num_instance x num_images_in_full_set, the R^2 scores between all unmasked or
    masked activations for one image, and all other unmasked activations of all other images in the full set.
    """

    # in case of pruned network
    if keep_dim is not None:
        repr_full_set = repr_full_set[:, keep_dim]

    corr_arr_list = []
    for i in range(repr_mat.shape[0]):
        corr_arr = []
        for j in range(repr_full_set.shape[0]):
            corr_arr.append(pearsonr(repr_mat[i], repr_full_set[j])[0])
        corr_arr = np.array(corr_arr)
        # delete the correlation between itself
        # because in the fMRI data there is no dis-similarity score between the images themselves
        corr_arr = np.delete(corr_arr, img_idx)
        corr_arr_list.append(corr_arr)

    return np.array(corr_arr_list)


def compute_dnn_sim(model, img_path, layer, repr_full_set, keep_dim, filter_size, stride, save_path):
    """
    1. Extract the activations of all masked version of the input image, then
    2. Compute the Pearson correlation between each instance in the collection of activations (unmasked and masked)
    and all other unmasked images.
    :param model: DNN pretrained model
    :param img_path: str, path to image
    :param layer: int, index of the layer that needed to extract (use print(net) to see the indices of layers)
    :param repr_full_set: numpy 2D array, activations of all original images from the full dataset
    :param keep_dim: list, list of indices of features that we want to keep. Only apply in the case of pruned network.
    :param filter_size: int, size in pixel of a square filter
    do not enter a 2D shape because the filter is square, later we construct the 2D size as filter_size x filter_size
    :param stride: int, distance in pixel between two consecutive filters
    :param save_path: str, path to save the results
    :return: numpy 2D array, shape num_instance x num_images_in_full_set, the R^2 scores between all unmasked or
    masked activations for one image, and all other unmasked activations of all other images in the full set.
    """

    # compute activations of all masked version of the input image
    repr_mat = extract_masked_repr(model=model, img_path=img_path, layer=layer,
                                   keep_dim=keep_dim, filter_size=filter_size, stride=stride)

    # figure out the index of image from its path
    img_idx = int(img_path.split('/')[-1].split('.')[0]) - 1

    # compute the Pearson correlation R^2 between each instance in the collection of activations (unmasked and masked)
    # and all other unmasked images.
    corr_arr = compute_corr_masked(repr_mat=repr_mat, repr_full_set=repr_full_set,
                                   keep_dim=keep_dim, img_idx=img_idx)
    if save_path:
        np.save(save_path, corr_arr)

    return corr_arr


def compute_heatmap(human_rdm, dnn_rdm, save_path=None):
    """
    Compute heatmap scores, which are Pearson correlation R^2 (2OI) between network's RDM and human's RDM
    :param human_rdm: numpy array, human RDM in 1D flattened format
    :param dnn_rdm: numpy 2D array, results of the function compute_dnn_sim
    :param save_path: str, path to save the heatmap scores
    :return: numpy array, list of heatmap scores
    """

    heatmap_scores_list = []
    for i in range(dnn_rdm.shape[0]):
        score = pearsonr(human_rdm, dnn_rdm[i])[0] ** 2
        heatmap_scores_list.append(score)
    heatmap_scores_list = np.array(heatmap_scores_list)

    if save_path:
        np.save(save_path, heatmap_scores_list)

    return heatmap_scores_list


if __name__ == '__main__':
    args = docopt(__doc__, version='Saliency map, ver 0.1')
    brain_area = args["--brain_area"]
    filter_size = int(args['--filter_size'])
    use_pruning = int(args['--use_pruning'])

    # # device = 'cpu'  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # # model = models.vgg19(weights='VGG19_Weights.DEFAULT')
    # # model = models.vgg19(pretrained=True)
    # # model.eval()
    #
    # dataset_name = 'set2' #  do not change to 2_1 anymore
    # stride = 4
    # color_grid_size = 4
    # # n_row = int((224 - filter_size) / stride) + 1
    # # n_col = int((224 - filter_size) / stride) + 1
    # n_row, n_col = 224 // stride, 224 // stride
    # # grid_size = 4
    #
    # repr_full_set = np.load(f'./data/{dataset_name}_repr_classifier_3.npy')
    # if use_pruning:
    #     keep_dim = np.load(f'./data/{dataset_name}_{brain_area}_pruned_feat.npy')
    # else:
    #     keep_dim = None
    # pathlib.Path(f'./res/{dataset_name}/score/{brain_area}/{filter_size}/{use_pruning}/').mkdir(parents=True, exist_ok=True)
    # pathlib.Path(f'./res/{dataset_name}/rdm/{brain_area}/{filter_size}/{use_pruning}/').mkdir(parents=True, exist_ok=True)
    #
    # img_path_list = ['./stimuli/set2/0017.jpg',
    #                  './stimuli/set2/0021.jpg',
    #                  # './stimuli/set2/0079.jpg',
    #                  # './stimuli/set2/0025.jpg',
    #                  # './stimuli/set2/0039.jpg',
    #                  # './stimuli/set2/0056.jpg',
    #                  './stimuli/set2/0109.jpg',
    #                  # './stimuli/set2/0111.jpg',
    #                  './stimuli/set2/0120.jpg',
    #                  # './stimuli/set2/0136.jpg',
    #                  # './stimuli/set2/0034.jpg',
    #                  './stimuli/set2/0036.jpg',
    #                  # './stimuli/set2/0061.jpg',
    #                  './stimuli/set2/0062.jpg',
    #                  # './stimuli/set2/0063.jpg',
    #                  # './stimuli/set2/0064.jpg',
    #                  './stimuli/set2/0065.jpg',
    #                  # './stimuli/set2/0066.jpg',
    #                  './stimuli/set2/0083.jpg',
    #                  # './stimuli/set2/0084.jpg',
    #                 ]
    # # img_path_list = sorted([f'./stimuli/{dataset_name}/' + i.name for i in pathlib.Path(f'./stimuli/set2_1/').glob('*')])
    # # img_path_list += sorted(
    # #     [f'./stimuli/{dataset_name}/' + i.name for i in pathlib.Path(f'./stimuli/set2_2/').glob('*')])
    #
    # # chosen_img_idx_list = sorted([int(i.name.split('.')[0]) for i in pathlib.Path(f'./stimuli/{dataset_name}/').glob('*')])
    # human_sim = get_human_sim_arr(brain_area=brain_area, dataset_name=dataset_name,
    #                               chosen_img_idx_list=[], restore_sim_mat=True)
    # # human_sim = np.array([[1.0, 1.0, 0.7, 0.4],
    # #                       [1.0, 1.0, 0.7, 0.4],
    # #                       [0.7, 0.7, 1.0, 0.7],
    # #                       [0.4, 0.4, 0.7, 1.0]])
    # # print(human_sim, human_sim.shape)
    #
    # for img_path in tqdm(img_path_list):
    #     # print('\n', img_path)
    #     img_idx = int(img_path.split('/')[-1].split('.')[0]) - 1
    #     # img_idx = img_path_list.index(img_path)
    #
    #     # dnn_sim = compute_dnn_sim(model=model, img_path=img_path, layer=3, repr_full_set=repr_full_set,
    #     #                           keep_dim=keep_dim, filter_size=filter_size, stride=stride,
    #     #                           save_path=f'./res/{dataset_name}/rdm/{brain_area}/{filter_size}/{use_pruning}/' + img_path.split('/')[-1].split('.')[0])
    #     dnn_sim = np.load(f'./res/{dataset_name}/rdm/{brain_area}/{filter_size}/{use_pruning}/' + img_path.split('/')[-1].split('.')[0] + '.npy')
    #     # dnn_sim = dnn_sim ** 2
    #     # print(dnn_sim, dnn_sim.shape)
    #     salient_scores = compute_saliency(human_sim=np.delete(human_sim[img_idx], img_idx), dnn_sim=1-dnn_sim,
    #             save_path=f'./res/{dataset_name}/score/{brain_area}/{filter_size}/{use_pruning}/' + img_path.split('/')[-1].split('.')[0])

    #################################################################################

    # brain_area = 'FFA'
    dataset_name = 'set2'
    # setting = 'large_unpruned'
    stride = 4
    color_grid_size = 4
    # n_row = int((224 - filter_size) / stride) + 1
    # n_col = int((224 - filter_size) / stride) + 1
    n_row, n_col = 224 // stride, 224 // stride

    # img_path_list = sorted([f'./stimuli/{dataset_name}/' + i.name for i in pathlib.Path(f'./stimuli/set2_2/').glob('*')])
    img_path_list = ['./stimuli/set2/0017.jpg',
                     './stimuli/set2/0021.jpg',
                     # './stimuli/set2/0079.jpg',
                     # './stimuli/set2/0025.jpg',
                     # './stimuli/set2/0039.jpg',
                     # './stimuli/set2/0056.jpg',
                     './stimuli/set2/0109.jpg',
                     # './stimuli/set2/0111.jpg',
                     './stimuli/set2/0120.jpg',
                     # './stimuli/set2/0136.jpg',
                     # './stimuli/set2/0034.jpg',
                     './stimuli/set2/0036.jpg',
                     # './stimuli/set2/0061.jpg',
                     './stimuli/set2/0062.jpg',
                     # './stimuli/set2/0063.jpg',
                     # './stimuli/set2/0064.jpg',
                     './stimuli/set2/0065.jpg',
                     # './stimuli/set2/0066.jpg',
                     './stimuli/set2/0083.jpg',
                     # './stimuli/set2/0084.jpg',
                     ]
    for img_path in tqdm(img_path_list):
        viz_scores = []
        for filter_size in range(24, 60, 4):
            salient_scores = np.load(f'./res/{dataset_name}/score/{brain_area}/{filter_size}/{use_pruning}/' + \
                                     img_path.split('/')[-1].split('.')[0] + '.npy')
            viz_scores.append(salient_scores[0] - salient_scores[1:].reshape([n_row, n_col]))
        viz_scores = np.array(viz_scores)
        # print(viz_scores.shape)
        _min, _max = viz_scores.min(axis=0), viz_scores.max(axis=0)
        viz_scores = _min * (np.abs(_min) > np.abs(_max)) + _max * (np.abs(_min) <= np.abs(_max))
        # viz_scores = viz_scores.mean(axis=0)
        # print(viz_scores.shape)
        pathlib.Path(f'./res/aggregate_absmax/{dataset_name}/img/{brain_area}/{use_pruning}/').mkdir(parents=True, exist_ok=True)
        show_heatmap(img_path=img_path, stride=stride, color_grid_size=color_grid_size,
                               salient_score=viz_scores, grayscale=True,
                               save_path=f'./res/aggregate_absmax/{dataset_name}/img/{brain_area}/{use_pruning}/' + img_path.split('/')[-1])
