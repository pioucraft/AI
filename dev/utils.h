#ifndef MAIN_H
#define MAIN_H

#define DATA_TYPE float
#define NUM_THREADS 256
#define TENSOR_MAX_RANK 4

#include <curand_kernel.h>

void checkCudaError();

typedef struct Convolution_Layer {
    int filter_dimensions;
    int filters_num;

    DATA_TYPE* filters;
    DATA_TYPE* biases;

    DATA_TYPE* filter_grads;
    DATA_TYPE* bias_grads;
} Convolution_Layer;

typedef struct Pooling_Layer {
    int pool_dimensions;
} Pooling_Layer;

typedef struct MLP_Layer {
    DATA_TYPE* weights;
    DATA_TYPE* biases;

    DATA_TYPE* weight_grads;
    DATA_TYPE* bias_grads;
} MLP_Layer;

typedef struct Dropout_Layer {
    DATA_TYPE dropout_rate;
    curandState_t* random_states;
    unsigned char* mask;
} Dropout_Layer;

typedef struct Layernorm_Layer {
    DATA_TYPE* gains;
    DATA_TYPE* biases;

    DATA_TYPE* gain_grads;
    DATA_TYPE* bias_grads;

    DATA_TYPE* means;
    DATA_TYPE* variances;

    DATA_TYPE* mean_grads;
    DATA_TYPE* variance_grads;

    DATA_TYPE* normalized_values;
} Layernorm_Layer;

typedef struct Softmax_Layer {
    DATA_TYPE temperature;
    DATA_TYPE* exp_values;
    DATA_TYPE* sums_exp_values;
    DATA_TYPE* grad_sums;
} Softmax_Layer;

typedef struct Weightstensormultiplication_Layer {
    DATA_TYPE* weights;
    DATA_TYPE* weight_grads;

    int weights_rank;
    DATA_TYPE weights_dimensions[TENSOR_MAX_RANK];
} Weightstensormultiplication_Layer;

typedef struct Layer {
    int layer_type;

    int num_in_channels;
    int num_out_channels;

    union {
        struct {
            int input_size;
            DATA_TYPE* input;
            DATA_TYPE* grads;
        } d1;
        struct {
            int input_dimensions;
            DATA_TYPE* input;
            DATA_TYPE* grads;
        } d2;
        struct {
            int tensor_rank;
            int tensor_dimensions[TENSOR_MAX_RANK];
            int input_size;

            DATA_TYPE* input;
            DATA_TYPE* grads;
        } tensor;
    } input;

    union {
        struct {
            int output_size;
            DATA_TYPE* output;
            DATA_TYPE* grads;
        } d1;
        struct {
            int output_dimensions;
            DATA_TYPE* output;
            DATA_TYPE* grads;
        } d2;
        struct {
            int tensor_rank;
            int tensor_dimensions[TENSOR_MAX_RANK];
            int output_size;

            DATA_TYPE* output;
            DATA_TYPE* grads;
        } tensor;
    } output;

    union {
        MLP_Layer mlp_layer;
        Pooling_Layer pooling_layer;
        Convolution_Layer convolution_layer;
        Dropout_Layer dropout_layer;
        Layernorm_Layer layernorm_layer;
        Softmax_Layer softmax_layer;
        Weightstensormultiplication_Layer weightstensormultiplication_layer;
    } layer;
} Layer;

typedef struct NN {
    int num_layers;
    Layer* layers;
} NN;

#endif
