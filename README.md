# Visualize perturbation scores by masking

## Install
We recommend installing the code in a virtual environment (under python 3.7). Create a virtual environment by yourself,
then run the command below in your terminal:

    git clone https://github.com/tlmnhut/Visualize_PrunedDNN_by_HumanSim.git
    cd Visualize_PrunedDNN_by_HumanSim/
    pip install -r requirements.txt

## Guild to reproduce the experiment
1. The stimuli and brain/behavior data were collected by [King et al., 2019](https://www.sciencedirect.com/science/article/pii/S1053811919303702).
Please download [their dataset](https://osf.io/ma2rj/) and place the fMRI data (FMRI.mat) in the folder *data*, and the
stimuli in the folder *stimuli* as below:

```
.
├── data
    ├── FMRI.mat
    ├── ...
├── stimuli
    ├── set1
    ├── set2
├── .gitignore
├── compute_perturbation.py
└── ...
```

2. Specify the path of images you want to use in *img_path_list* in **compute_perturbation.py**.
3. Obtain the perturbation scores by running:

```
python compute_perturbation.py --brain_area PPA --filter_size 56 --use_pruning 1
```

in which brain_area is PPA, FFA, or vTC; filter_size is the size of the mask in pixel; and use_pruning is 1 if we 
use pruned features, 0 if use unpruned features.

4. Plot the perturbation scores on top of original image by running:

```
python visualize.py
```
    
You will find the result image in ./res folder.

5. That is just one image, one RDM from one brain area, one filter size, and only pruned network. To fully reproduce
all results for many images, 3 brain areas PPA FFA vTC, 9 filter size from 24, 28, ... to 56, both pruned and unpruned
network, run:

```
chmod u+x run.sh
./run.sh
```


It will take really long time to finish (days).

## Role of each script

- **compute_perturbation.py**: compute the perturbation scores
- **visualize.py**: plot heatmap
- **run.sh**: to reproduce the whole experiments (take long time to run)
- **pruning.py**: prune network
- **extract_feature.py**: extract features of stimuli
- **utils.py**: utility and helper functions

## Sample results

Please check folder *results* to see the sample results.
