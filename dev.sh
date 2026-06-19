#!/bin/sh
echo "Choose what to run:"
echo "1) MNIST script (normal)"
echo "2) Language script"
printf "Enter 1 or 2: "
read choice

echo "Building the project..."
case "$choice" in
  1)
    nvcc -o main dev/utils.cu dev/nn.cu dev/mlp.cu dev/relu.cu dev/tanh.cu dev/pooling.cu dev/convolution.cu dev/mnist.cu dev/dropout.cu mnist/mnist.cu --threads 4
    ;;
  2)
    nvcc -o main dev/utils.cu dev/nn.cu dev/mlp.cu dev/relu.cu dev/tanh.cu language/language.cu dev/language.cu --threads 4
    ;;
  *)
    echo "Invalid choice. Exiting."
    exit 1
    ;;
esac

echo "Running the project..."
> test_accuracy.data
time ./main
