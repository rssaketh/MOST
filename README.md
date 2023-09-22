# Official Implementation of MOST (ICCV 2023 Oral)
[Sai Saketh Rambhatla](https://rssaketh.github.io), [Ishan Misra](https://imisra.github.io), [Rama Chellappa](https://engineering.jhu.edu/faculty/rama-chellappa/), [Abhinav Shrivastava](https://www.cs.umd.edu/~abhinav/)

![alt text](https://github.com/rssaketh/MOST/blob/main/assets/teaser_nohuman.png?raw=true)

This is the official repository for the work **MOST**:**M**ultiple **O**bject localization using **S**elf-supervised **T**ransformers for object discovery, accepted as an Oral at ICCV 2023, see [Project Page](rssaketh.github.io/most).

## Introduction
We tackle the challenging task of unsupervised object localization in this work. Recently, transformers trained with self-supervised learning have been shown to exhibit object localization properties without being trained for this task. In this work, we present Multiple Object localization with Self-supervised Transformers (MOST) that uses features of transformers trained using self-supervised learning to localize multiple objects in real world images. MOST analyzes the similarity maps of the features using box counting; a fractal analysis tool to identify tokens lying on foreground patches. The identified tokens are then clustered together, and tokens of each cluster are used to generate bounding boxes on foreground regions. Unlike recent state-of-the-art object localization methods, MOST can localize multiple objects per image and outperforms SOTA algorithms on several object localization and discovery benchmarks on PASCAL-VOC 07, 12 and COCO20k datasets. Additionally, we show that MOST can be used for self-supervised pre-training of object detectors, and yields consistent improvements on fully, semi-supervised object detection and unsupervised region proposal generation.


## Installation instructions
- This code was tested using Python3.7

We recommend using conda to create a new environment.

```
conda create -n most python=3.7
```

Then activate the virtual environment

```
conda activate most
```

Install Pytorch 1.7.1 (CUDA 10.2)

```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
```

- Install other requirements

```
pip install -r requirements.txt
```
- Install DINO
```
git clone https://github.com/facebookresearch/dino.git
cd dino; 
touch __init__.py
echo -e "import sys\nfrom os.path import dirname, join\nsys.path.insert(0, join(dirname(__file__), '.'))" >> __init__.py; cd ../;
```
## MOST on a single image
To apply MOST to an example image, run the following example
```
python main_most.py --image_path <path_to_image> --visualize pred
```
Results are stored in the output directory given by parameter `output_dir`.

## Run MOST on datasets
To run MOST on PASCAL VOC, COCO or other custom datasets, follow the dataset instructions of [LOST](https://github.com/valeoai/LOST#launching-lost-on-datasets).
To launch MOST on PASCAL VOC 2007 and 2012 datasets, run
```
python main_most.py --dataset VOC07 --set trainval
python main_most.py --dataset VOC12 --set trainval
```
For COCO dataset, run
```
python main_most.py --dataset COCO20k --set train
```

To run with different patch sizes or architectures, run
```
python main_most.py --dataset VOC07 --set trainval #VIT-S/16
python main_most.py --dataset VOC07 --set trainval --patch_size 8 #VIT-S/8
python main_most.py --dataset VOC07 --set trainval --arch vit_base #VIT-B/16
```
