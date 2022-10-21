import numpy as np
import pathlib
import cv2
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torchvision.transforms as transforms


def create_img_with_heatmap(img_path, stride, color_grid_size, salient_score, grayscale=True, save_path=None):
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


def plot_heatmap(img_path_list, dataset_name, brain_area, use_pruning, aggregate_method):
    """
    Compute the 2OI_baseline - 2OI_masked, aggregate results from multiple filter size, then produce the heatmaps.
    Parameters
    ----------
    img_path_list: list, list of image paths
    dataset_name: str, either 'set1' or 'set2' as in King et al.'s dataset
    You can create your own set of data, and change the name of the dataset here
    brain_area: str, either PPA, FFA, vTC
    use_pruning: int, either 0 (unpruned) or 1 (pruned)
    aggregate_method: str, either mean or absmax
    For each filter size, we obtain a matrix of scores. To plot the heatmap, we have to collapse many matrices of scores
    into 1 matrix only.
    Mean: take the average of all score matrices. Absmax: take the absolute value first, then take max.

    Returns
    -------
    None
    """

    stride = 4
    color_grid_size = 4
    n_row, n_col = 224 // stride, 224 // stride

    for img_path in tqdm(img_path_list):
        viz_scores = []
        for filter_size in range(24, 60, 4):  # we have filter size from 24, 28 ... to 56
            try:
                perturbation_scores = np.load(f'./res/{dataset_name}/score/{brain_area}/{filter_size}/{use_pruning}/' + \
                                              img_path.split('/')[-1].split('.')[0] + '.npy')
                # 2OI_baseline - 2OI_masked
                viz_scores.append(perturbation_scores[0] - perturbation_scores[1:].reshape([n_row, n_col]))
            except FileNotFoundError:
                pass
        viz_scores = np.array(viz_scores)
        # viz_scores = viz_scores * ((viz_scores <= -0.01)*1 + (viz_scores >= 0.01)*1)  # TODO

        # aggregate by taking mean or absmax
        if aggregate_method == 'absmax':
            _min, _max = viz_scores.min(axis=0), viz_scores.max(axis=0)
            viz_scores = _min * (np.abs(_min) > np.abs(_max)) + _max * (np.abs(_min) <= np.abs(_max))
        elif aggregate_method == 'mean':
            viz_scores = viz_scores.mean(axis=0)
        else:
            raise

        # produce heatmap
        pathlib.Path(f'./res/aggregate_{aggregate_method}/{dataset_name}/img/{brain_area}/{use_pruning}/').mkdir(
            parents=True, exist_ok=True)
        create_img_with_heatmap(img_path=img_path, stride=stride, color_grid_size=color_grid_size,
            salient_score=viz_scores, grayscale=True,
            save_path=f'./res/aggregate_{aggregate_method}/{dataset_name}/img/{brain_area}/{use_pruning}/' +
                      img_path.split('/')[-1])


def show_img_grid(img_path_list, aggregate_method):
    """
    Plot result images in grid
    :param img_path_list: list, list of image paths
    :param aggregate_method: str, either mean or absmax
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

    pathlib.Path(f'./res/grid/set2_{aggregate_method}/').mkdir(parents=True, exist_ok=True)

    f, axarr = plt.subplots(1, 5, figsize=(20, 6), gridspec_kw={'wspace':0.05, 'hspace':0.05})
    for img_path in tqdm(img_path_list):
        axarr[0].imshow(_resize_crop_img(img_path))
        # axarr[0, 0].imshow(Image.open(f'./res/aggregate_{aggregate_method}/set2/img/FFA/0/' + img_path.split('/')[-1]))
        # axarr[1, 0].imshow(Image.open(f'./res/aggregate_{aggregate_method}/set2/img/FFA/1/' + img_path.split('/')[-1]))
        axarr[1].imshow(Image.open(f'./res/aggregate_{aggregate_method}/set2/img/PPA/0/' + img_path.split('/')[-1]))
        axarr[2].imshow(Image.open(f'./res/aggregate_{aggregate_method}/set2/img/PPA/1/' + img_path.split('/')[-1]))
        axarr[3].imshow(Image.open(f'./res/aggregate_{aggregate_method}/set2/img/vTC/0/' + img_path.split('/')[-1]))
        axarr[4].imshow(Image.open(f'./res/aggregate_{aggregate_method}/set2/img/vTC/1/' + img_path.split('/')[-1]))

        # labels
        label_font_size = 20
        axarr[0].set_xlabel('Original image', fontsize=label_font_size)

        # axarr[0, 0].xaxis.set_label_position('top')
        # axarr[0, 0].set_ylabel('full set')
        # axarr[1, 0].set_xlabel('FFA pruned', fontsize=label_font_size)
        # axarr[0, 1].xaxis.set_label_position('top')
        axarr[1].set_xlabel('PPA unpruned', fontsize=label_font_size)
        axarr[2].set_xlabel('PPA pruned', fontsize=label_font_size)
        axarr[3].set_xlabel('vTC unpruned', fontsize=label_font_size)
        axarr[4].set_xlabel('vTC pruned', fontsize=label_font_size)
        # for position in range(0, 5): axarr[position].xaxis.set_label_position('top')

        # Turn off tick labels
        for position in [0, 1, 2, 3, 4]:
            axarr[position].set_xticks([])
            axarr[position].set_yticks([])

        # add boxes
        for position in [1, 2, 3, 4]:
            axarr[position].add_patch(Rectangle((27, 27), 170, 170, linewidth=1, edgecolor='black', facecolor='none'))

        # # Some special treatments for border
        # for position in ['top', 'bottom', 'right', 'left']:
        #     axarr[1, 0].spines[position].set_visible(False)

        plt.savefig(f'./res/grid/set2_{aggregate_method}/' + img_path.split('/')[-1], bbox_inches='tight')


if __name__ == '__main__':
    img_path_list = [
        './stimuli/set2/0017.jpg',
        # './stimuli/set2/0021.jpg',
        # './stimuli/set2/0079.jpg',
        # './stimuli/set2/0025.jpg',
        # './stimuli/set2/0039.jpg',
        # './stimuli/set2/0056.jpg',
        # './stimuli/set2/0109.jpg',
        # './stimuli/set2/0111.jpg',
        # './stimuli/set2/0120.jpg',
        # './stimuli/set2/0136.jpg',
        # './stimuli/set2/0034.jpg',
        # './stimuli/set2/0036.jpg',
        # './stimuli/set2/0061.jpg',
        # './stimuli/set2/0062.jpg',
        # './stimuli/set2/0063.jpg',
        # './stimuli/set2/0064.jpg',
        # './stimuli/set2/0065.jpg',
        # './stimuli/set2/0066.jpg',
        # './stimuli/set2/0083.jpg',
        # './stimuli/set2/0084.jpg',
        #
        # # texture
        # './stimuli/set2/0002.jpg',
        # './stimuli/set2/0004.jpg',
        # './stimuli/set2/0006.jpg',
        # './stimuli/set2/0012.jpg',
        # './stimuli/set2/0013.jpg',
        # './stimuli/set2/0018.jpg',
        # './stimuli/set2/0019.jpg',
        # './stimuli/set2/0020.jpg',
        # './stimuli/set2/0027.jpg',
        # './stimuli/set2/0050.jpg',
        # './stimuli/set2/0055.jpg',
        # './stimuli/set2/0075.jpg',
        # './stimuli/set2/0080.jpg',
        # './stimuli/set2/0081.jpg',
        # './stimuli/set2/0090.jpg',
        # './stimuli/set2/0093.jpg',
        # './stimuli/set2/0106.jpg',
        # './stimuli/set2/0138.jpg',
        #
        # # imagenet class
        # './stimuli/set2/0001.jpg',
        # './stimuli/set2/0014.jpg',
        # './stimuli/set2/0029.jpg',
        # './stimuli/set2/0033.jpg',
        # './stimuli/set2/0077.jpg',
        # './stimuli/set2/0089.jpg',
        # './stimuli/set2/0112.jpg',
        # './stimuli/set2/0114.jpg',
        # './stimuli/set2/0118.jpg',
        # './stimuli/set2/0119.jpg',
        # './stimuli/set2/0121.jpg',
        # './stimuli/set2/0144.jpg',
    ]

    for brain_area in ['PPA', 'vTC']:
        for use_pruning in [0, 1]:
            for aggregate_method in ['mean', 'absmax']:
                plot_heatmap(img_path_list=img_path_list,
                             dataset_name='set2',
                             brain_area=brain_area,
                             use_pruning=use_pruning,
                             aggregate_method=aggregate_method)

    for aggregate_method in ['mean', 'absmax']:
        show_img_grid(img_path_list=img_path_list, aggregate_method=aggregate_method)
