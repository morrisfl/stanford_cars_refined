# Refined Stanford Cars Dataset
![](readme/sc_img_example.jpeg)
**Figure 1:** *Example images from the Stanford Cars [1] dataset. Showing cars from the same model which previously 
belong to the same class. After the refinement, each column correspond to an individual class.*

This repository contains a refined annotation file for the Stanford Cars [1] dataset, characterized by an increased 
class granularity. In its original form, the dataset contained 196 classes, with each class denoting a different car model. 
After the refinement process, the dataset now contains 1,600 classes, with each class representing a unique combination 
of car model and color, as shown in Figure 1.

The refinement process involved leveraging the color information inherent in the images. Accordingly, various car color 
classification models were subjected to fine-tuning (FT) or linear probing (LP) and evaluation on the Vehicle Color Recognition 
(VCoR) dataset [2]. These models were then used to predict the colors of cars in the Stanford Cars [1] dataset. 
The predicted color information was then used to increase the class granularity of the Stanford Cars dataset.

The repository also contains the code used for training the color classification models (see [IV. Training](#iv-training)), 
as well as the code used for the refinement process (see [V. Refinement](#v-refinement)). The refined annotation file 
of the Stanford Cars dataset is provided in the `data` directory.

## I. Setup

Here, we describe a step-by-step guide to setup and install dependencies on a UNIX-based system, such as Ubuntu, using 
`conda` as package manager. If `conda` is not available, alternative package managers such as `venv` can be used.

#### 1. Create a virtual environment
```
conda create -n env_sc_refine python=3.8
conda activate env_sc_refine
```
#### 2. Clone the repository
```
git clone git@github.com:morrisfl/stanford_cars_refined.git
```
#### 3. Install pytorch
Depending on your system and compute requirements, you may need to change the command below. See [pytorch.org](https://pytorch.org/get-started/locally/) 
for more details.
```
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```
#### 4. Install the repository with all dependencies
```
cd stanford_cars_refined
python -m pip install .
```
If you want to make changes to the code, you can install the repository in editable mode:
```
python -m pip install -e .
```

## III. Data Preparation

### VCoR Dataset
During the training process, the VCoR [2] dataset was used. The dataset consists of approximately 10,500 images distributed 
over 15 different car color classes and is divided into training (7.5k images), validation (1.5k images), and test (1.5k images) 
sets. Notably, only 10 of the 15 classes were used for training, and classes such as *beige*, *gold*, *pink*, *purple*, 
and *tan* were excluded from the process. The rationale for excluding these classes was based on the understanding that 
these colors do not predominantly represent colors associated with cars in the Stanford Cars [1] dataset.
The VCoR dataset can be downloaded from [here](https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset)
and should be placed in a `<data_dir>` directory. The directory structure should look like this:
```
<data_dir>
└───vcor-vehicle-color-recognition-dataset
    ├───test
    ├───train
    └───val
```

### Stanford Cars Dataset
The training set of the Stanford Cars [1] dataset was used for the refinement process. The training set consists of
8,144 images distributed over 196 different car model classes. The dataset can be downloaded from [here](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)
and the corresponding annotation file in csv utilized for the refinement process can be downloaded from [here](https://github.com/BotechEngineering/StanfordCarsDatasetCSV/tree/main).
The images and annotation file should be placed in a `<data_dir>` directory. The directory structure should look like this:
```
<data_dir>
└───stanford-cars-dataset
    ├───cars_train/cars_train
    └───train.csv
```

## IV. Training
We trained (LP, FT, and LP-FT) different color classification models on the VCoR [2] dataset. The training settings and 
results are shown below.

### Linear Probing (LP)
Run the following command to linear probe a model on the VCoR [2] dataset:
```
python src/train.py siglip <data_dir> \
    --output_dir results/ \
    --batch_size 64 \
    --train_style lp \
    --epochs 10 \
    --optimizer adamw \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler cosine \
    --min_lr 1e-4 \
    --warmup_steps 82 \
    --warmup_factor 0.1 \
```

| Model           | Pre-training dataset | Validation Accuracy | Test Accuracy |
|-----------------|----------------------|:-------------------:|:-------------:|
| SigLIP ViT-B/16 | WebLI                |       0.9281        |    0.9105     |
| DINOv2 ViT-B/16 | LVD-142M             |       0.7511        |    0.7350     |
| CLIP ConvNeXt-B | LAION-2B             |       0.9173        |    0.9167     |
| ConvNeXt-B      | ImageNet-1k          |       0.8419        |    0.8308     |
| EfficientNet-B1 | ImageNet-1k          |       0.7170        |    0.6580     |
**Table 1:** *Linear probing results on the VCoR [2] dataset.*

### Fine-tuning (FT)
Run the following command to fine-tune a model on the VCoR [2] dataset:
```
python src/train.py siglip <data_dir> \
    --output_dir results/ \
    --batch_size 64 \
    --train_style ft \
    --epochs 10 \
    --optimizer adamw \
    --lr lr \
    --weight_decay 1e-4 \
    --scheduler cosine \
    --min_lr min_lr \
    --warmup_steps 82 \
    --warmup_factor 0.1 \
```

| Model           | Pre-training dataset | `lr` | `min_lr` | Validation Accuracy | Test Accuracy |
|-----------------|----------------------|:----:|:--------:|:-------------------:|:-------------:|

**Table 2:** *Fine-tuning results on the VCoR [2] dataset.*

### Linear Probing + Fine-tuning (LP-FT)
Run the following command to linear probe and fine-tune a model on the VCoR [2] dataset:
```
python src/train.py siglip <data_dir> \
    --output_dir results/ \
    --batch_size 64 \
    --train_style lp-ft \
    --epochs 10 \
    --optimizer adamw \
    --lr lr \
    --ft_lr_factor 1e-2 \
    --weight_decay 1e-4 \
    --scheduler cosine \
    --min_lr min_lr \
    --warmup_steps 82 \
    --warmup_factor 0.1 \
```

| Model           | Pre-training dataset | `lr` | `min_lr` | Validation Accuracy | Test Accuracy |
|-----------------|----------------------|:----:|:--------:|:-------------------:|:-------------:|

**Table 3:** *Linear probing + fine-tuning results on the VCoR [2] dataset.*

## V. Refinement



## References
[1] Krause, Jonathan, et al. "3d object representations for fine-grained categorization." Proceedings of the IEEE 
International Conference on Computer Vision. 2013.

[2] Dincer, Berkay, and Cemal Köse. "Vehicle color recognition using deep learning." 2019 27th Signal Processing and 
Communications Applications Conference (SIU). IEEE, 2019.

