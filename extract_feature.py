import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


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


if __name__ == '__main__':
    device = 'cpu'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg19(pretrained=True)
    model.eval()
    # print(model)

    # default preprocess for VGG-19
    dataset = datasets.ImageFolder(
        # You have to create a subfolder named as a class (e.g. '0') to contain all images within set 2
        # See more about it here: https://debuggercafe.com/pytorch-imagefolder-for-training-cnn-models/
        root='./stimuli_/set2',  # Replace the input image folder here
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

    data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    feats = extract_feat(net=model, layer=3, dataloader=data_loader, device=device)
    np.save('./data/set2_repr_classifier_3', feats)  # Replace the output file name here
    print('Features/activations are extracted and saved in ./data')
