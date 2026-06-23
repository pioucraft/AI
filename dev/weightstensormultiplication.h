#ifndef WEIGHTSTENSORMULTIPLICATION_H
#define WEIGHTSTENSORMULTIPLICATION_H

#include <cuda_runtime.h>
#include <stdio.h>

#include "nn.h"
#include "utils.h"

int create_weightstensormultiplication(Layer* layer, int tensor_rank, int tensor_dimensions[TENSOR_MAX_RANK], int weights_rank, int weights_dimensions[TENSOR_MAX_RANK]);

/*
__global__ void softmax_zero_exp_sums(Layer layer);
__global__ void softmax_compute_exps(Layer layer);
__global__ void softmax_compute_outputs(Layer layer);


__global__ void zero_input_grads_softmax_layer(Layer layer);

__global__ void grad_softmax_layer(Layer layer);

__global__ void grad_softmax_layer_step_1(Layer layer);
__global__ void grad_softmax_layer_step_2(Layer layer);
*/

#endif
