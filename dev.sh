#!/bin/sh
default_temp=$2
default_chars=$3
echo "Choose what to run:"
echo "1) MNIST script (normal)"
echo "2) Language script (training)"
echo "3) Language inference [temp] [chars]"
printf "Enter 1-3: "
read line
set -- $line
choice=$1
temp=${2:-$default_temp}
chars=${3:-$default_chars}

echo "Building the project..."
case "$choice" in
  1)
    nvcc -o main dev/utils.cu dev/nn.cu dev/mlp.cu dev/relu.cu dev/tanh.cu dev/pooling.cu dev/convolution.cu dev/mnist.cu dev/dropout.cu dev/layernorm.cu mnist/mnist.cu dev/softmax.cu dev/attention.cu --threads 4
    ;;
  2)
    nvcc -o main dev/utils.cu dev/nn.cu dev/mlp.cu dev/relu.cu dev/gelu.cu dev/tanh.cu dev/pooling.cu dev/convolution.cu dev/dropout.cu dev/layernorm.cu language/language.cu dev/language.cu dev/softmax.cu dev/attention.cu --threads 4
    ;;
  3)
    nvcc -o main dev/utils.cu dev/nn.cu dev/mlp.cu dev/relu.cu dev/gelu.cu dev/tanh.cu dev/pooling.cu dev/convolution.cu dev/dropout.cu dev/layernorm.cu language/language.cu dev/language-inference.cu dev/softmax.cu dev/attention.cu --threads 4
    ;;
  *)
    echo "Invalid choice. Exiting."
    exit 1
    ;;
esac

echo "Running the project..."
> test_accuracy.data
time ./main $temp $chars
