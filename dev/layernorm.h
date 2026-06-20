#ifndef LAYERNORM_H
#define LAYERNORM_H

#include <cuda_runtime.h>

#include "nn.h"
#include "utils.h"

int create_layernorm_layer(Layer* layer, int input_size);

__global__ void layernorm_forward(Layer layer);
__global__ void layernorm_forward_mean(Layer layer);
__global__ void layernorm_forward_variance(Layer layer);

__global__ void zero_input_grads_layernorm_layer(Layer layer);

__global__ void grad_layernorm_layer(Layer layer);

#endif
