# Relation Networks for Object Detection


The major contributors of this repository include [Dazhi Cheng](https://github.com/chengdazhi), [Jiayuan Gu](https://github.com/Jiayuan-Gu), [Han Hu](https://github.com/ancientmooner) and [Zheng Zhang](https://github.com/stupidZZ).


## Introduction

**Relation Networks for Object Detection** is described in an [CVPR 2018 oral paper](https://arxiv.org/abs/1711.11575). 

## Disclaimer

This is an official implementation for [Relation Networks for Object Detection](https://arxiv.org/abs/1711.11575) based on MXNet. It is worth noting that:

  * This repository is tested on official [MXNet v1.1.0@(commit 629bb6)](https://github.com/apache/incubator-mxnet/commit/e29bb6f76365e45dd44e23941692c9d969959315). You should be able to use it with any version of MXNET that contains required operators like Deformable Convolution. 
  * We trained our model based on the ImageNet pre-trained [ResNet-v1-101](https://github.com/KaimingHe/deep-residual-networks) using a [model converter](https://github.com/dmlc/mxnet/tree/430ea7bfbbda67d993996d81c7fd44d3a20ef846/tools/caffe_converter). The converted model produces slightly lower accuracy (Top-1 Error on ImageNet val: 24.0% v.s. 23.6%).
  * This repository is based on [Deformable ConvNets](https://github.com/msracver/Deformable-ConvNets).

## License

© Microsoft, 2018. Licensed under an MIT license.

## Citing Relation Networks for Object Detection

If you find Relation Networks for Object Detection useful in your research, please consider citing:
```
@article{hu2017relation,
  title={Relation Networks for Object Detection},
  author={Hu, Han and Gu, Jiayuan and Zhang, Zheng and Dai, Jifeng and Wei, Yichen},
  journal={arXiv preprint arXiv:1711.11575},
  year={2017}
} 
```

## Main Results

#### Faster RCNN

|                                 | <sub>training data</sub> | <sub>testing data</sub>  | <sub>mAP</sub>  | <sub>mAP@0.5</sub> | <sub>mAP@0.75</sub>| <sub>mAP@S</sub> | <sub>mAP@M</sub> | <sub>mAP@L</sub> | <sub>Inference Time</sub> | <sub>Post Processing Time</sub> |
|---------------------------------|---------------|---------------|------|---------|---------|-------|-------|-------|---------------------------------|---------------------------------|
| <sub>2FC + nms(0.5)<br /> ResNet-101 </sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 31.8 | 53.9 | 32.2 | 10.5 | 35.2 | 51.5 | 0.168s | 0.025s |
| <sub>2FC + softnms(0.6)<br /> ResNet-101 </sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 32.3 | 52.8 | 34.1 | 11.1 | 35.9 | 51.8 | 0.200s | 0.060s |
| <sub>2FC + Relation Module + softnms<br />ResNet-101</sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 34.7 | 55.3 | 37.2 | 13.7 | 38.8 | 53.6 | 0.211s | 0.059s |
| <sub>2FC + Learn NMS </br>ResNet-101 </sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 32.6 | 51.8 |  35.0  | 11.8 | 36.6 | 52.1 | 0.162s | 0.020s |
| <sub>2FC + Relation Module + Learn NMS(e2e)<br />ResNet-101</sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 35.2 | 55.5 | 38.0 | 15.2 | 39.2 | 54.1 | 0.175s | 0.022s |

#### Deformable Faster RCNN

|                                 | <sub>training data</sub> | <sub>testing data</sub>  | <sub>mAP</sub>  | <sub>mAP@0.5</sub> | <sub>mAP@0.75</sub>| <sub>mAP@S</sub> | <sub>mAP@M</sub> | <sub>mAP@L</sub> | <sub>Inference Time</sub> | <sub>NMS Time</sub> |
|---------------------------------|---------------|---------------|------|---------|---------|-------|-------|-------|---------------------------------|---------------------------------|
| <sub>2FC + nms(0.5)<br /> ResNet-101 </sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 37.2 | 58.1 | 40.0 | 16.4 | 41.3 | 55.5 | 0.180s | 0.022s |
| <sub>2FC + softnms(0.6)<br /> ResNet-101 </sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 37.5 | 57.3 | 41.0 | 16.6 | 41.7 | 55.8 | 0.208s | 0.052s |
| <sub>2FC + Relation Module + Learn NMS(e2e)<br /> ResNet-101 </sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 38.4 | 57.6 | 41.6 | 18.2 | 43.1 | 56.6 | 0.188s | 0.023s |

#### FPN

|                                 | <sub>training data</sub> | <sub>testing data</sub>  | <sub>mAP</sub>  | <sub>mAP@0.5</sub> | <sub>mAP@0.75</sub>| <sub>mAP@S</sub> | <sub>mAP@M</sub> | <sub>mAP@L</sub> | <sub>Inference Time</sub> | <sub>NMS Time</sub> |
|---------------------------------|---------------|---------------|------|---------|---------|-------|-------|-------|---------------------------------|---------------------------------|
| <sub>2FC + nms(0.5)<br /> ResNet-101 </sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 36.6 | 59.3 | 39.3 | 20.3 | 40.5 | 49.4 | 0.196s | 0.037s |
| <sub>2FC + softnms(0.6)<br /> ResNet-101 </sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 36.8 | 57.8 | 40.7 | 20.4 | 40.8 | 49.7 | 0.323s | 0.167s |
| <sub>2FC + Relation Module + Learn NMS(e2e)<br /> ResNet-101 </sub> | <sub>coco trainval35k</sub> | <sub>coco minival</sub> | 38.6 | 59.9 | 43.0 | 22.1 | 42.3 | 52.8 | 0.232s | 0.022s |


*Running time is counted on a single Maxwell Titan X GPU (mini-batch size is 1 in inference).*

## Requirements: Software

1. MXNet from [the offical repository](https://github.com/apache/incubator-mxnet). We tested our code on [MXNet v1.1.0@(commit 629bb6)](https://github.com/apache/incubator-mxnet/commit/e29bb6f76365e45dd44e23941692c9d969959315). Due to the rapid development of MXNet, it is recommended to checkout this version if you encounter any issues. We may maintain this repository periodically if MXNet adds important feature in future release.

2. Python 2.7. We recommend using Anaconda2 as it already includes many common packages. We do not support Python 3 yet, if you want to use Python 3 you need to modify the code to make it work.


3. The following Python packages:
  ```
  Cython
  EasyDict
  mxnet-cu80
  opencv-python >= 3.2.0
  ```


## Requirements: Hardware

Any NVIDIA GPUs with at least 6GB memory should be OK.

## Installation

1. Clone the Relation Networks for Object Detection repository.
```
git clone https://github.com/msracver/Relation-Networks-for-Object-Detection.git
cd Relation-Networks-for-Object-Detection
```

2. Run `sh ./init.sh`. The scripts will build cython module automatically and create some folders.

3. Install MXNet:

  ***Quick start***

  3.1 Install MXNet and all dependencies by 
  ```
  pip install -r requirements.txt
  ```
  If there is no other error message, MXNet should be installed successfully. 

  ***Build from source (alternative way)***

  3.2 Clone MXNet v1.1.0 by
  ```
  git clone -b v1.1.0 --recursive https://github.com/apache/incubator-mxnet.git
  ```
  3.3 Compile MXNet
  ```
  cd ${MXNET_ROOT}
  make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
  ```
  3.4 Install the MXNet Python binding by

  ***Note: If you will actively switch between different versions of MXNet, please follow 3.5 instead of 3.4***
  ```
  cd python
  sudo python setup.py install
  ```
  3.5 For advanced users, you may put your Python packge into `./external/mxnet/$(YOUR_MXNET_PACKAGE)/mxnet`, and modify `MXNET_VERSION` in `./experiments/relation_rcnn/cfgs/*.yaml` to `$(YOUR_MXNET_PACKAGE)`. Thus you can switch among different versions of MXNet quickly.

## Preparation for Training & Testing

1. Please download COCO datasets, and make sure it looks like this:

  ```
  ./data/coco/
  ```

2. Please download ImageNet-pretrained ResNet-v1-101 backbone model and Faster RCNN ResNet-v1-101 model manually from [OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqpCxvNTMZDlcDTpSA) or [Baiduyun (password:pech)](https://pan.baidu.com/s/1GMca4yxLoQMV2tfBabWQ3Q), and put it under folder `./model/pretrained_model`. Make sure it looks like this:
  ```
  ./model/pretrained_model/resnet_v1_101-0000.params
  ```
  We use a pretrained Faster RCNN and fix its params when training Faster RCNN with Learn NMS head. If you are trying to conduct such experiments, please also include the pretrained Faster RCNN model from OneDrive and put it under folder `./model/pretrained_model`. Make sure it looks like this:

  ```
  ./model/pretrained_model/coco_resnet_v1_101_rcnn-0008.params
  ```

3. For FPN related experiments, we use proposals generated by a pretrained RPN to speed up our experiments. Please download the proposals from [OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqpEnDg8s4FH33zh8g) or Baiduyun (due to its size constraint) [part1 (password:g24u)](https://pan.baidu.com/s/1Wr54mf54G2URnLW-A9bGFA) [part2 (password:ipa8)](https://pan.baidu.com/s/1RNggxprIPxNb1S6aQ9hiZw) and put it under folder `./proposal/resnet_v1_101_fpn/rpn_data`. Make sure it looks like this:

   ```
   ./proposal/resnet_v1_101_fpn/rpn_data/COCO_minival2014_rpn.pkl
   ./proposal/resnet_v1_101_fpn/rpn_data/COCO_train2014_rpn.pkl
   ./proposal/resnet_v1_101_fpn/rpn_data/COCO_valminusminival2014_rpn.pkl
   ```

## Demo Models

We provide trained relation network models, covering all settings in the above Main Results table.

1. To try out our pre-trained relation network models, please download manually from [OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqpD-UHVYNbj25lU0w) or [Baiduyun (password:9q6i)](https://pan.baidu.com/s/1DBZmbpBxn4NaqY8ljLdEUg), and put it under folder `output/`.

	Make sure it looks like this:
	```
	./output/rcnn/coco/resnet_v1_101_coco_trainvalminus_rcnn_end2end_8epoch/train2014_valminusminival2014/rcnn_coco-0008.params
	./output/rcnn/coco/resnet_v1_101_coco_trainvalminus_rcnn_end2end_relation_8epoch/train2014_valminusminival2014/rcnn_coco-0008.params
	./output/rcnn/coco/resnet_v1_101_coco_trainvalminus_rcnn_end2end_learn_nms_3epoch/train2014_valminusminival2014/rcnn_coco-0003.params
	./output/rcnn/coco/resnet_v1_101_coco_trainvalminus_rcnn_end2end_relation_learn_nms_8epoch/train2014_valminusminival2014/rcnn_coco-0008.params
	./output/rcnn/coco/resnet_v1_101_coco_trainvalminus_rcnn_dcn_end2end_8epoch/train2014_valminusminival2014/rcnn_coco-0008.params
	./output/rcnn/coco/resnet_v1_101_coco_trainvalminus_rcnn_dcn_end2end_relation_learn_nms_8epoch/train2014_valminusminival2014/rcnn_coco-0008.params
	./output/rcnn/coco/resnet_v1_101_coco_trainvalminus_rcnn_fpn_8epoch/train2014_valminusminival2014/rcnn_fpn_coco-0008.params
	./output/rcnn/coco/resnet_v1_101_coco_trainvalminus_rcnn_fpn_relation_learn_nms_8epoch/train2014_valminusminival2014/rcnn_fpn_coco-0008.params
	```
2. To run the Faster RCNN with Relation Module and Learn NMS model, run
	```
	python experiments/relation_rcnn/rcnn_test.py --cfg experiments/relation_rcnn/cfgs/resnet_v1_101_coco_trainvalminus_rcnn_end2end_relation_learn_nms_8epoch.yaml --ignore_cache
	```
	If you want to try other models, just change the config files. There are ten config files in `./experiments/relation_rcnn/cfg` folder, eight of which are provided with pretrained models.


## Usage

1. All of our experiment settings (GPU #, dataset, etc.) are kept in yaml config files at folder `./experiments/relation_rcnn/cfgs`.

2. Ten config files have been provided so far, namely, Faster RCNN, Deformable Faster RCNN and FPN with 2FC head, 2FC + Relation Head and 2FC + Relation + Learn NMS(e2e), and an additional Faster RCNN with 2FC + Learn NMS head. We use 4 GPUs to train our models.

3. To perform experiments, run the python scripts with the corresponding config file as input. For example, to train and test Faster RCNN with Relation Module and Learn NMS(e2e), use the following command:
    ```
    python experiments\relation_rcnn\rcnn_end2end_train_test.py --cfg experiments/relation_rcnn/cfgs/resnet_v1_101_coco_trainvalminus_rcnn_end2end_relation_learn_nms_8epoch.yaml
    ```
    A cache folder would be created automatically to save the model and the log under `output/rcnn/`.

    The rcnn_end2end_train_test.py script is for Faster RCNN and Deformable Faster RCNN experiments that train RPN together with RCNN. To train and test FPN which use previously generated proposals, use the following command:

    ```
    python experiments\relation_rcnn\rcnn_train_test.py --cfg experiments/relation_rcnn/cfgs/resnet_v1_101_coco_trainvalminus_fpn_relation_learn_nms_8epoch.yaml
    ```

4. Please find more details in config files and in our code.

## FAQ

Q: I encounter `segment fault` at the beginning.

A: A compatibility issue has been identified between MXNet and opencv-python 3.0+. We suggest that you always `import cv2` first before `import mxnet` in the entry script. 

<br/>

