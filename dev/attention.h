#ifndef ATTENTION_H
#define ATTENTION_H

#include <cuda_runtime.h>
#include <stdio.h>

#include "nn.h"
#include "utils.h"

int create_attention_layer(Layer* layer, int context_length, int embedding_size, int query_key_size, int num_heads);

__global__ void attention_forward_key_query(Layer layer);

__global__ void attention_forward_key_query(Layer layer);
__global__ void attention_forward_value(Layer layer);
__global__ void attention_forward_scores(Layer layer);
__global__ void attention_forward_masking(Layer layer);
__global__ void attention_softmax_zero_exp_sums(Layer layer);
__global__ void attention_softmax_compute_exps(Layer layer);
__global__ void attention_softmax_compute_outputs(Layer layer);
__global__ void attention_forward_value_weighted_sum(Layer layer);

/*
__global__ void zero_input_grads_layernorm_layer(Layer layer);

__global__ void grad_layernorm_layer(Layer layer);
__global__ void grad_layernorm_layer_step_two(Layer layer);

__global__ void zero_grads_layernorm_layer(Layer layer);

int save_layernorm_layer(Layer layer, FILE* file);
int load_layernorm_layer(Layer* layer, FILE* file);
*/

#endif
