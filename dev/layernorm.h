#ifndef LAYERNORM_H
#define LAYERNORM_H

#include <cuda_runtime.h>
#include <stdio.h>

#include "nn.h"
#include "utils.h"

int create_layernorm_layer(Layer* layer, int tensor_rank, int tensor_dimensions[TENSOR_MAX_RANK]);

__global__ void layernorm_forward_zero_variance_mean(Layer layer);
__global__ void layernorm_forward(Layer layer);
__global__ void layernorm_forward_mean(Layer layer);
__global__ void layernorm_forward_variance(Layer layer);

__global__ void zero_input_grads_layernorm_layer(Layer layer);

__global__ void grad_layernorm_layer(Layer layer);
__global__ void grad_layernorm_layer_step_two(Layer layer);

__global__ void zero_grads_layernorm_layer(Layer layer);

__global__ void update_layernorm_layer(Layer layer, DATA_TYPE learning_rate, int timestep, DATA_TYPE weight_decay);

int save_layernorm_layer(Layer layer, FILE* file);
int load_layernorm_layer(Layer* layer, FILE* file);

#endif
