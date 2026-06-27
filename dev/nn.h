#ifndef NN_H
#define NN_H

#define LAYER_TYPE_MLP 1
#define LAYER_TYPE_POOLING 2
#define LAYER_TYPE_CONVOLUTION 3
#define LAYER_TYPE_RELU 4
#define LAYER_TYPE_TANH 5
#define LAYER_TYPE_DROPOUT 6
#define LAYER_TYPE_LAYERNORM 7
#define LAYER_TYPE_SOFTMAX 8
#define LAYER_TYPE_ATTENTION 9
#define LAYER_TYPE_GELU 10

#include "utils.h"

int create_nn(NN* nn);

int call_nn(NN* nn, DATA_TYPE* input, int run_dropout);

int zero_grads_nn(NN* nn);

int grad_nn(NN* nn, DATA_TYPE* expected_output);

int clip_grads_nn(NN* nn, DATA_TYPE max_norm);

int update_nn(NN* nn, DATA_TYPE learning_rate, DATA_TYPE weight_decay);

int save_nn(NN* nn, const char* filename);

int load_nn(NN* nn, const char* filename);

__global__ void add_pos_encoding(DATA_TYPE* output, DATA_TYPE* pos_encoding, int total_size);
__global__ void grad_pos_encoding(DATA_TYPE* pos_grads, DATA_TYPE* grads, int total_size);
__global__ void update_pos_encoding(DATA_TYPE* pos, DATA_TYPE* pos_grads, AdamW_State adam,
                                    int total_size, DATA_TYPE lr, int timestep, DATA_TYPE wd);
__global__ void add_residual(DATA_TYPE* dst, DATA_TYPE* src, int total_size);
__global__ void add_residual_grad(DATA_TYPE* dst_grad, DATA_TYPE* src_grad, int total_size);

#endif
