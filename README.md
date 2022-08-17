# SimpleNet-CIFAR

Implementation of a convolutional neural network, SimpleNet in Python using Pytorch. CIFAR-100 dataset is used for training and testing the model. CIFAR-100 has 100 classes, and each class contains 600 images. The dataset contains 32x32 color images. For this implementation (SimpleNet), the images are resized to 3x227x227.

## Implement the basic network, train it for 10 epochs.

Summary of network: 

```---------------------------------------------------------------- 
Layer (type)        Output            Shape Param
================================================================
Conv2d-1        [-1, 64, 114, 114]      1,792 
MaxPool2d-2     [-1, 64, 57, 57]          0 
Conv2d-3        [-1, 128, 29, 29]       73,856
MaxPool2d-4     [-1, 128, 15, 15]         0
Conv2d-5        [-1, 256, 8, 8]         295,168
MaxPool2d-6     [-1, 256, 4, 4]           0
Linear-7        [-1, 1024]              4,195,328
Linear-8        [-1, 1024]              1,049,600
Linear-9        [-1, 100]               102,500
================================================================
Total params: 5,718,244
Trainable params: 5,718,244
Non-trainable params: 0
---------------------------------------------------------------- 
Input size (MB): 0.59
Forward/backward pass size (MB): 9.15
Params size (MB): 21.81
Estimated Total Size (MB): 31.55
----------------------------------------------------------------```
