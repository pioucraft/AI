#!/bin/sh
echo "Building the project..."
nvcc -o main dev/main.cu dev/utils.cu dev/mnist.cu dev/nn.cu dev/mlp.cu dev/pooling.cu dev/convolution.cu dev/relu.cu dev/tanh.cu dev/dropout.cu
echo "Running the project..."
> test_accuracy.data
time ./main
