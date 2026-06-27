#ifndef GELU_H
#define GELU_H

#include <cuda_runtime.h>

#include "nn.h"
#include "utils.h"

int create_gelu_layer(Layer* layer, int input_size);

__global__ void gelu_forward(Layer layer);

__global__ void zero_input_grads_gelu_layer(Layer layer);

__global__ void grad_gelu_layer(Layer layer);

#endif
