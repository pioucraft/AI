#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <cuda_runtime.h>
#include <stdio.h>

#include "nn.h"
#include "utils.h"

int create_softmax_layer(Layer* layer, int tensor_rank, int tensor_dimensions[TENSOR_MAX_RANK], DATA_TYPE temperature);

__global__ void softmax_zero_exp_sums(Layer layer);
__global__ void softmax_compute_max(Layer layer);
__global__ void softmax_compute_exps(Layer layer);
__global__ void softmax_compute_outputs(Layer layer);


__global__ void zero_input_grads_softmax_layer(Layer layer);

__global__ void grad_softmax_layer(Layer layer);

__global__ void grad_softmax_layer_step_1(Layer layer);
__global__ void grad_softmax_layer_step_2(Layer layer);

__global__ void grad_softmax_simplified(Layer layer, DATA_TYPE* expected_output);
__global__ void grad_softmax_simplified_scaled(Layer layer, DATA_TYPE* expected_output, DATA_TYPE scale);

#endif
