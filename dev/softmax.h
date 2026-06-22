#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <cuda_runtime.h>
#include <stdio.h>

#include "nn.h"
#include "utils.h"

int create_softmax_layer(Layer* layer, int tensor_rank, int tensor_dimensions[TENSOR_MAX_RANK], DATA_TYPE temperature);

/*
__global__ void layernorm_forward_zero_variance_mean(Layer layer);
__global__ void layernorm_forward(Layer layer);
__global__ void layernorm_forward_mean(Layer layer);
__global__ void layernorm_forward_variance(Layer layer);

__global__ void zero_input_grads_layernorm_layer(Layer layer);

__global__ void grad_layernorm_layer(Layer layer);
__global__ void grad_layernorm_layer_step_two(Layer layer);

__global__ void zero_grads_layernorm_layer(Layer layer);

int save_layernorm_layer(Layer layer, FILE* file);
int load_layernorm_layer(Layer* layer, FILE* file);
*/

#endif
