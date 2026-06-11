#ifndef NN_H
#define NN_H

#define LAYER_TYPE_MLP 1
#define LAYER_TYPE_POOLING 2
#define LAYER_TYPE_CONVOLUTION 3
#define LAYER_TYPE_RELU 4
#define LAYER_TYPE_TANH 5
#define LAYER_TYPE_DROPOUT 6

#include "utils.h"

int create_nn(NN* nn);

int call_nn(NN* nn, DATA_TYPE* input, int run_dropout);

int zero_grads_nn(NN* nn);

int grad_nn(NN* nn, DATA_TYPE* expected_output);

int update_nn(NN* nn, DATA_TYPE learning_rate);

int save_nn(NN* nn, const char* filename);

int load_nn(NN* nn, const char* filename);

#endif
