import numpy as np
from scipy.io import loadmat
from PIL import Image
import torch
import torchvision.transforms as transforms


def upper_tri(r):
    """
    Computes and returns the off-diagonal upper triangle of a square matrix
    as a vector
    Parameters
    ----------
    r: (N, N) ndarray
          N by N matrix for which the upper traingle elements are to be returned
    Returns
    ---------
    r_offdiag : ndarray
          A vector of length N*(N-1)/2
    Note: this function trims the lower triangle part of the similarity matrix,
    and keep the upper triangle part
    """

    # Extract off-diagonal elements of each Matrix
    ioffdiag = np.triu_indices(r.shape[0], k=1)  # indices of off-diagonal elements
    r_offdiag = r[ioffdiag]
    return r_offdiag


def get_activations(net, layer, input_tensor, device, keep_dim=None):
    """
    Extract the activations from a specific layer of either pruned or unpruned network
    :param net: DNN pretrained model
    :param layer: int, index of the layer that needed to extract (use print(net) to see the indices of layers)
    :param input_tensor: tensor, shape n_sample x n_feature_input
    :param device: str, either 'cpu' or 'cuda'
    :param keep_dim: list, in case of pruned network, this is the retained list of index of features
    :return: tensor, shape n_sample x n_feature
    """

    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    net._modules['classifier'][layer].register_forward_hook(get_features('feat'))

    feats, labels = [], []
    # loop through batches
    with torch.no_grad():
        # print(path)
        # each epoch, select 16 samples only to save memory
        for i in range(0, input_tensor.shape[0], 16):
            outputs = net(input_tensor[i:i+16].to(device))
            feats.append(features['feat'].detach().cpu().numpy())
            # labels.append(label.numpy())

    feats = np.concatenate(feats)
    # labels = np.concatenate(labels)

    if keep_dim is not None:
        feats = feats[:, keep_dim]

    return feats


def extract_masked_repr(model, img_path, layer, keep_dim, filter_size, stride, device='cpu'):
    """
    Compute activations of all masked version of the input image.
    Logic of this function: 1. Sweep through the input image as we usually do with Convolutional neural network,
    at each position we masked the image by a zero filter; 2. Pass the original and the collection of masked images
    through the DNN model to extract the activations from a specific layer.
    :param model: DNN pretrained model
    :param img_path: path to the input image
    :param layer: int, index of the layer that needed to extract (use print(net) to see the indices of layers)
    :param keep_dim: list, in case of pruned network, this is the retained list of index of features
    :param filter_size: int, size in pixel of a square filter
    do not enter a 2D shape because the filter is square, later we construct the 2D size as filter_size x filter_size
    :param stride: int, distance in pixel between two consecutive filters
    :param device: str, either 'cpu' or 'cuda'
    :return: tensor, shape n_masked_image + 1 x n_features, + 1 means we also keep the activations of the unmasked image
    A collection of activations of all masked versions of the original image. The first row also contains the
    activation of the unmasked original image.
    """

    # Define of resize and add zero padding to image
    input_size = 224  # standard size of VGG-19 input images
    padding_size = (filter_size - stride) // 2
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # standard VGG-19 normalizing
        transforms.Pad(padding_size),
    ])

    # read and preprocess image
    input_image = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(input_image)

    n_row, n_col = 224 // stride, 224 // stride
    img_collection = [input_tensor]  # first row contains original unmasked image
    # Sweep through the image as we do with Convolutional neural network
    for r in range(n_row):
        for c in range(n_col):
            masked_img = input_tensor.detach().clone()
            # zero masking
            masked_img[:, padding_size + r*stride:padding_size + r*stride + filter_size,
                          padding_size + c*stride:padding_size + c*stride + filter_size] = 0
            img_collection.append(masked_img)

    # Extract activations from the collection of original and masked images
    output = get_activations(net=model, layer=layer, input_tensor=torch.stack(img_collection),
                             device=device, keep_dim=keep_dim)
    return output


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
