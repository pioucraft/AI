#ifndef DROPOUT_H
#define DROPOUT_H

#include <cuda_runtime.h>

#include "nn.h"
#include "utils.h"

int create_dropout_layer(Layer* layer, int input_size, DATA_TYPE dropout_rate);

__global__ void dropout_forward(Layer layer, int run_dropout);

__global__ void zero_input_grads_dropout_layer(Layer layer);

__global__ void grad_dropout_layer(Layer layer);

#endif
