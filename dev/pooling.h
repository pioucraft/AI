#ifndef POOLING_H 
#define POOLING_H

#include <cuda_runtime.h>

#include "nn.h"
#include "utils.h"

int create_pooling_layer(Layer* layer, int input_dimensions, int output_dimensions, int pool_dimensions, int channels);

__global__ void pooling_forward(Layer layer);

__global__ void zero_input_grads_pooling_layer(Layer layer);

__global__ void grad_pooling_layer(Layer layer);

#endif
