#ifndef RELU_H 
#define RELU_H 

#include <cuda_runtime.h>

#include "nn.h"
#include "utils.h"

int create_relu_layer(Layer* layer, int input_size);

__global__ void relu_forward(Layer layer);

__global__ void zero_input_grads_relu_layer(Layer layer);

__global__ void grad_relu_layer(Layer layer);

#endif
