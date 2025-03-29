# Object Detector for Autonomous Vehicles Based on Faster RCNN 
Zirui Wang

- Pytorch implementation of Faster R-CNN based on VGG16.

## 1. Introduction

![architecture](./images/model_architecture.png)

This is a implementation a framework to Faster RCNN on object detection tasks.The whole model is 
implemented on Pytorch and trained on KITTI 2d Object Detection 
dataset, training on KITTI 2D object detection training set and evaluate on validation set.

## 2. Experimental Results

- Detection results on KITTI 2d Object Detection valication set
  - All models were evaluated using COCO-style detection evaluation metrics.

| Training dataset |          Model           |   mAP@[.5,.95]  |   mAP@[.75,.95]  |
| :--------------: | :----------------------: | :-------------: | :--------------: |
|     KITTI 2d     |        Faster RCNN       |      71.58      |      32.40       | 

## 3. Requirements

- numpy
- six
- torch
- torchvision
- tqdm
- cv2
- defaultdict
- itertools
- namedtuple
- skimage
- xml
- pascal_voc_writer
- PIL

## 4. Usage

### 4.1 Data preparation

- Download the training, validation, and test data.

```shell
# KITTI 2d Object Detection training set and groundtruth labels
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
```

- Untar files into directy named `Kitti`

```shell
# KITTI 2d Object Detection trainset and labels (following last command)
cd ..
mkdir Kitti && cd Kitti
unzip data_object_image_2.zip
unzip data_object_label_2.zip
```

- The KITTI dataset need to reformat to match with the following structure. 

```shell
dataset
   ├── Kitti
   │   ├── training
   │       ├── image_2
   │       └── label_2
```

- Convert the KITTI dataset into PASCAL VOC 2007 dataset format using the dataset format convertion tool script

```python
from data.kitti2voc import kitti2voc
kitti_dataset_dir = os.path.abspath("/data/cloner174/Kitti")
voc_dataset_dir = os.path.abspath("/data/cloner174/KITTI2VOC")
train_ratio = 0.8
class_names=['Person_sitting',"Pedestrian",'Cyclist',"Truck","Car","Tram","Van"]
kitti2voc(kitti_dataset_dir,voc_dataset_dir,train_ratio,class_names)
```

- After running the above command, you should have the same dataset structure for KITTI as VOC 2007, and it is now 
ready to load into the model

```shell
dataset
   ├── KITTI2VOC
   │   ├── Annotations
   │   ├── ImageSets
   │   ├── JPEGImages
```

### 4.2 Train models

- You can easily modify the parameters for training in `utils/config.py` and run the following script for model training

```shell
# if you are in local environemnt, run:
python3 ./train.py
# if you are in conda environment, run: 
python ./train.py
```

### 4.3 Test models

- You can easily modify the parameters for testing in `utils/config.py` and run the following script for model training
- You can visualize the testing image by setting `visualize=True` in configuration file
- The output image should be placed under `save_dir/visuals` specified in the configuration file

- Ex 1) VGG16 (file name: "vgg16_1.pth")

```shell
# if you are in local environemnt, run:
python3 ./test.py
# if you are in conda environment, run: 
python ./test.py
```

### 4.4 Example output images

- Below are some of the resulting images from visualization

<p align="center">
  <img src="./images/output3.jpg">
</p>
<p align="center">
  <img src="./images/output6.jpg">
</p>
<p align="center">
  <img src="./images/output8.jpg">
</p>

## 5. Reference

- [simple-faster-rcnn](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)
