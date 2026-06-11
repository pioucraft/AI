#ifndef TANH_H 
#define TANH_H 

#include <cuda_runtime.h>

#include "nn.h"
#include "utils.h"

int create_tanh_layer(Layer* layer, int input_size);

__global__ void tanh_forward(Layer layer);

__global__ void zero_input_grads_tanh_layer(Layer layer);

__global__ void grad_tanh_layer(Layer layer);

#endif
