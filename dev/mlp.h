#ifndef MLP_H
#define MLP_H

#include <cuda_runtime.h>
#include <stdio.h>

#include "nn.h"
#include "utils.h"

int create_mlp_layer(Layer* layer, int input_size, int output_size);

__global__ void mlp_forward(Layer layer);

__global__ void zero_grads_mlp_layer(Layer layer);

__global__ void grad_mlp_layer(Layer layer);

__global__ void update_mlp_layer(Layer layer, DATA_TYPE learning_rate);

__global__ void zero_input_grads_mlp_layer(Layer layer);

int save_mlp_layer(Layer layer, FILE* file);

int load_mlp_layer(Layer* layer, FILE* file);

#endif
