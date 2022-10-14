# Enabling ISP-less Low-Power Computer Vision WACV2023

## Framework
Our experiments are based on the [mmfewshot](https://github.com/open-mmlab/mmfewshot) framework, plesase follow its instruction to install mmfewshot and mmdetection.

## Raw Dataset
Our released dataset can be download from this [link](https://drive.google.com/file/d/1gNG-GVd5gb-PO8U4hDkO_EL138S7e3MO/view?usp=sharing). It consists of 123287 npy files with a total size of 391 GB.

## Training setup
To integrate our dataset with mmfewshot, you should use our functions (```mmdet/datasets/pipelines```) to load the dataset.

Our configurations for base training and few-shot learning by using Faster RCNN model are provided in ```configs/detection/tfa/coco```.

Our test dataset is [PASCALRAW](https://searchworks.stanford.edu/view/hq050zr7488), we apply the few-shot learning on the dataset containing the 10X downscaled images.
You need to generate a class-blanced subset of the dataset as the training dataset, and use the rest as the test dataset.

Note, since we applied the ```demosaic``` function to the raw dataset, we need to divide the bbox by 2 in the annotation files for few-shot learning.