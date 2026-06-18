#!/bin/sh
echo "Building the project..."
nvcc -o main dev/main.cu dev/utils.cu dev/nn.cu dev/mlp.cu dev/pooling.cu dev/convolution.cu dev/relu.cu dev/tanh.cu dev/dropout.cu language/language.cu --threads 4
echo "Running the project..."
> test_accuracy.data
time ./main
