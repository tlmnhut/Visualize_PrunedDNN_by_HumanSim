import numpy as np
from tqdm import tqdm
import os
import pathlib
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import cv2


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


def extract_feat(net, layer, dataloader, device):
    """
    Extract the activations from a specific layer from a pretrained network
    :param net: DNN pretrained model
    :param layer: int, index of the layer that needed to extract (use print(net) to see the indices of layers)
    :param dataloader: pytorch data loader
    :param device: str, either 'cpu' or 'cuda'
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
        for images, label in dataloader:
            # print(path)
            outputs = net(images.to(device))
            feats.append(features['feat'].detach().cpu().numpy())
            labels.append(label.numpy())

    feats = np.concatenate(feats)
    # labels = np.concatenate(labels)

    return feats.reshape(len(dataloader.dataset), -1)


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


def show_heatmap(img_path, stride, color_grid_size, salient_score, grayscale=True, save_path=None):
    """
    Transform salient scores to colors and display them on top of the image.
    Green: area that if masked, the 2OI score decreases compared to unmasked 2OI,
    Red: area that if masked, the 2OI score increases compared to unmasked 2OI.
    :param img_path: str, path to image
    :param stride: int, distance in pixel between two consecutive filters
    :param color_grid_size: int, size in pixel of a square colored region. Do not enter a 2D shape
    because the color grid is square, later we construct the 2D size as color_grid_size x color_grid_size.
    :param salient_score: matrix, salient scores for each colored region
    :param grayscale: bool, if True then use gray image background, otherwise keep the color original background
    :param save_path: str, path to save the heat map image
    :return: None, if save_path is provided then the heatmap will be saved, otherwise just display the heatmap
    """

    input_size = 224
    preprocess_m = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor()
    ])

    # background image
    input_image = Image.open(img_path)
    if grayscale:
        input_image = input_image.convert('L')
    else:
        input_image = input_image.convert('RGB')
    input_image = preprocess_m(input_image)
    input_image = input_image.numpy()
    input_image = 255.0 / input_image.max() * (input_image - input_image.min())  # scale to display
    if grayscale:
        # because the heatmap is RGB, we have to convert the background again to RGB but the colors are still gray
        input_image = np.repeat(input_image[0][None, ...], 3, axis=0)

    # create the heatmap based on the salient scores
    # first create a 0-value image, then assign the values to each region to create colors
    mask_tensor = torch.zeros([3, input_size, input_size])
    n_row, n_col = 224 // stride, 224 // stride
    for r in range(n_row):
        for c in range(n_col):
            color_region = torch.zeros([3, color_grid_size, color_grid_size])
            if salient_score[r, c] > 0:  # 2OI masked decrease compared to 2OI baseline, color green
                try:
                    # scale the shade of color based on the max score
                    color_region[1] = int(salient_score[r, c] / np.max(salient_score) * 255)
                except ValueError:
                    color_region[1] = 0
            else:  # 2OI masked increase, color red
                try:
                    # scale the shade of color based on the min score
                    color_region[0] = int(salient_score[r, c] / np.min(salient_score) * 255)
                except ValueError:
                    color_region[0] = 0
            coor_row = r * stride
            coor_col = c * stride
            # assign color
            mask_tensor[:, coor_row:coor_row + color_grid_size, coor_col:coor_col + color_grid_size] = color_region
    mask = mask_tensor.numpy()

    # put the mask color image on top of background image
    img_with_mask = cv2.addWeighted(mask, 0.3, input_image, 0.7, 0, mask)

    # post-processing
    rescaled = (255.0 / img_with_mask.max() * (img_with_mask - img_with_mask.min())).astype(np.uint8)
    im = Image.fromarray(rescaled.transpose(1, 2, 0), 'RGB')
    if save_path:
        im.save(save_path)
    else:
        im.show()


def show_img_grid():
    """
    Plot result images in grid
    :return: None
    """

    def _resize_crop_img(img_path):
        """
        Resize and crop original image to fit their size with other images.
        :param img_path: str, path to original image
        :return: Image
        """
        preprocess_m = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        image = Image.open(img_path).convert('RGB')
        image = preprocess_m(image)
        return image

    aggregate_method = 'mean'

    pathlib.Path(f'./res/grid/set2_{aggregate_method}/').mkdir(parents=True, exist_ok=True)
    # img_path_list = sorted([i.name for i in pathlib.Path(f'./stimuli/set2_2/').glob('*')])
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

    f, axarr = plt.subplots(2, 3, figsize=(12, 9))
    for img_path in tqdm(img_path_list):
        # axarr[0, 0].imshow(_resize_crop_img(f'./stimuli/set2_2/' + img_path))
        axarr[0, 0].imshow(Image.open(f'./res/aggregate_{aggregate_method}/set2/img/FFA/0/' + img_path.split('/')[-1]))
        axarr[1, 0].imshow(Image.open(f'./res/aggregate_{aggregate_method}/set2/img/FFA/1/' + img_path.split('/')[-1]))
        axarr[0, 1].imshow(Image.open(f'./res/aggregate_{aggregate_method}/set2/img/PPA/0/' + img_path.split('/')[-1]))
        axarr[1, 1].imshow(Image.open(f'./res/aggregate_{aggregate_method}/set2/img/PPA/1/' + img_path.split('/')[-1]))
        axarr[0, 2].imshow(Image.open(f'./res/aggregate_{aggregate_method}/set2/img/vTC/0/' + img_path.split('/')[-1]))
        axarr[1, 2].imshow(Image.open(f'./res/aggregate_{aggregate_method}/set2/img/vTC/1/' + img_path.split('/')[-1]))

        # labels
        label_font_size = 20
        axarr[0, 0].set_xlabel('FFA unpruned', fontsize=label_font_size)
        # axarr[0, 0].xaxis.set_label_position('top')
        # axarr[0, 0].set_ylabel('full set')
        axarr[1, 0].set_xlabel('FFA pruned', fontsize=label_font_size)
        # axarr[0, 1].xaxis.set_label_position('top')
        axarr[0, 1].set_xlabel('PPA unpruned', fontsize=label_font_size)
        axarr[1, 1].set_xlabel('PPA pruned', fontsize=label_font_size)
        axarr[0, 2].set_xlabel('vTC unpruned', fontsize=label_font_size)
        axarr[1, 2].set_xlabel('vTC pruned', fontsize=label_font_size)

        # Turn off tick labels
        axarr[0, 0].set_xticks([])
        axarr[0, 0].set_yticks([])
        axarr[0, 1].set_xticks([])
        axarr[0, 1].set_yticks([])
        axarr[1, 0].set_xticks([])
        axarr[1, 0].set_yticks([])
        axarr[1, 1].set_xticks([])
        axarr[1, 1].set_yticks([])
        axarr[0, 2].set_xticks([])
        axarr[0, 2].set_yticks([])
        axarr[1, 2].set_xticks([])
        axarr[1, 2].set_yticks([])

        plt.savefig(f'./res/grid/set2_{aggregate_method}/' + img_path.split('/')[-1])


if __name__ == '__main__':
    # device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = models.vgg19(pretrained=True)
    # model.eval()
    # #print(model)
    # # img_path_list = ['stimuli/set1/0002.jpg']
    # #
    # # extract_masked_repr(model=model, img_path_list=img_path_list,
    # #                     layer=[4], filter_size=int(224/2), stride=int(224/2))
    #
    # dataset = datasets.ImageFolder(
    #     root='./stimuli_/set_test_2',
    #     transform=transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ]))
    # data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    # tmp = extract_feat(net=model, layer=3, dataloader=data_loader, device=device)
    # np.save('./data/set_test_2_repr_classifier_3', tmp)

    # repr_mat = np.load('./data/set2_repr_classifier_3.npy')
    # df = pd.DataFrame(repr_mat.T)
    # print(df)
    # corr_mat = df.corr()
    # print(corr_mat, corr_mat.shape)
    # np.save('./data/set2_corr', corr_mat)

    show_img_grid()

