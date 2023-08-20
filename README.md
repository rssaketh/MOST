# Official Implementation of MOST (ICCV 2023 Oral)
[Sai Saketh Rambhatla](https://rssaketh@github.io), [Ishan Misra](https://imisra.github.io), [Rama Chellappa](https://engineering.jhu.edu/faculty/rama-chellappa/), [Abhinav Shrivastava](https://abhinavsh.info)

![alt text](https://github.com/rssaketh/MOST/blob/main/assets/teaser_nohuman.png?raw=true)

This is the official repository for the work **MOST**:**M**ultiple **O**bject localization using **S**elf-supervised **T**ransformers for object discovery, accepted as an $${\color{red}ORAL}$$ at ICCV 2023, see [Project Page](rssaketh.github.io/most).

## Introduction
We tackle the challenging task of unsupervised object localization in this work. Recently, transformers trained with self-supervised learning have been shown to exhibit object localization properties without being trained for this task. In this work, we present Multiple Object localization with Self-supervised Transformers (MOST) that uses features of transformers trained using self-supervised learning to localize multiple objects in real world images. MOST analyzes the similarity maps of the features using box counting; a fractal analysis tool to identify tokens lying on foreground patches. The identified tokens are then clustered together, and tokens of each cluster are used to generate bounding boxes on foreground regions. Unlike recent state-of-the-art object localization methods, MOST can localize multiple objects per image and outperforms SOTA algorithms on several object localization and discovery benchmarks on PASCAL-VOC 07, 12 and COCO20k datasets. Additionally, we show that MOST can be used for self-supervised pre-training of object detectors, and yields consistent improvements on fully, semi-supervised object detection and unsupervised region proposal generation.


## Installation instructions
- This code was tested using Python3.7
We recommend using conda to create a new environment.
```conda create -n most python=3.7```
Then activate the virtual environment
```conda activate most```
Install Pytorch 1.7.1 (CUDA 10.2)
``` conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch ```
- Install other requirements
```pip install -r requirements.txt```
