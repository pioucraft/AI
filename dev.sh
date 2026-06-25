#!/bin/sh
echo "Choose what to run:"
echo "1) MNIST script (normal)"
echo "2) Language script"
printf "Enter 1 or 2: "
read choice

echo "Building the project..."
case "$choice" in
  1)
    nvcc -o main dev/utils.cu dev/nn.cu dev/mlp.cu dev/relu.cu dev/tanh.cu dev/pooling.cu dev/convolution.cu dev/mnist.cu dev/dropout.cu dev/layernorm.cu mnist/mnist.cu dev/softmax.cu dev/attention.cu --threads 4
    ;;
  2)
    nvcc -o main dev/utils.cu dev/nn.cu dev/mlp.cu dev/relu.cu dev/tanh.cu dev/pooling.cu dev/convolution.cu dev/dropout.cu dev/layernorm.cu language/language.cu dev/language.cu dev/softmax.cu dev/attention.cu --threads 4
    ;;
  *)
    echo "Invalid choice. Exiting."
    exit 1
    ;;
esac

echo "Running the project..."
> test_accuracy.data
time ./main
