# Min-Max-Convolutional-Neural-Networks

A Pytorch implementation of Min-Max Convolutional Neural Networks for Image Classification. It is implemented for MNIST dataset and CIFAR-10 dataset. Implemented a Custom MIN-MAX Convolution Block.

# Model Architecture.

Used a LeNet type Architecture with Conv->Pool->Conv->Pool->Conv->Pool->FC->FC(OUTPUT LAYER)

## For MNIST

For MNIST dataset I used 3 Convolutional Layers each of 64 filters, kernel size 5,padding 2,stride 1 followed by MaxPooling Layers of size (3,3) and stride 2.

## For CIFAR-10

For CIFAR-10 dataset I used 2 Convolutional Layers each of 32 filters and another Convolutional Layer of 64 filters, kernel size 5,padding 2,stride 1 followed by MaxPooling Layers of size (3,3) and stride 2.


# Requirements

```
PyTorch>=1.0.0

TorchVision
```

## How to Run

```
git clone https://github.com/avinashsai/Min-Max-Convolutional-Neural-Networks.git
```

 To run MINMAX CNN for MNIST:
 
 ```
 cd MNIST
 
 python min_max_cnn_mnist.py
 ```
 
 To run MINMAX CNN for MNIST:
 
 ```
 cd CIFAR-10
 
 python min_max_cnn_cifar.py
 ````
