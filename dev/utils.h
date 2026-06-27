#ifndef MAIN_H
#define MAIN_H

#define DATA_TYPE float
#define NUM_THREADS 256
#define TENSOR_MAX_RANK 4

#define ADAMW_BETA1 0.9f
#define ADAMW_BETA2 0.999f
#define ADAMW_EPSILON 1e-8f

#include <curand_kernel.h>

void checkCudaError();
int check_nan(DATA_TYPE* buffer, int size, const char* name);
DATA_TYPE normal_sample(DATA_TYPE mean, DATA_TYPE std);

typedef struct AdamW_State {
    DATA_TYPE* m;
    DATA_TYPE* v;
} AdamW_State;

typedef struct Convolution_Layer {
    int filter_dimensions;
    int filters_num;

    DATA_TYPE* filters;
    DATA_TYPE* biases;

    DATA_TYPE* filter_grads;
    DATA_TYPE* bias_grads;

    AdamW_State filters_adam;
    AdamW_State biases_adam;
} Convolution_Layer;

typedef struct Pooling_Layer {
    int pool_dimensions;
} Pooling_Layer;

typedef struct MLP_Layer {
    DATA_TYPE* weights;
    DATA_TYPE* biases;

    DATA_TYPE* weight_grads;
    DATA_TYPE* bias_grads;

    AdamW_State weights_adam;
    AdamW_State biases_adam;
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

    AdamW_State gains_adam;
    AdamW_State biases_adam;

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

typedef struct Attention_Layer {
    int context_length;
    int embedding_size;
    int query_key_size;
    int num_heads;

    DATA_TYPE* query_weights;
    DATA_TYPE* key_weights;
    DATA_TYPE* value_weights;

    DATA_TYPE* query_weight_grads;
    DATA_TYPE* key_weight_grads;
    DATA_TYPE* value_weight_grads;

    AdamW_State query_adam;
    AdamW_State key_adam;
    AdamW_State value_adam;

    DATA_TYPE* queries;
    DATA_TYPE* keys;
    DATA_TYPE* values;

    DATA_TYPE* attention_scores;
    DATA_TYPE* attention_score_grads;

    DATA_TYPE* attention_scores_masked;
    DATA_TYPE* attention_score_masked_grads;

    DATA_TYPE* softmax_exp_values;
    DATA_TYPE* softmax_sums_exp_values;
    DATA_TYPE* softmax_grad_sums;

    DATA_TYPE* attention_percentages;
    DATA_TYPE* attention_percentage_grads;

    DATA_TYPE* value_grads;
    DATA_TYPE* query_grads;
    DATA_TYPE* key_grads;

    DATA_TYPE* out_weight;
    DATA_TYPE* out_bias;
    DATA_TYPE* out_weight_grads;
    DATA_TYPE* out_bias_grads;
    AdamW_State out_weight_adam;
    AdamW_State out_bias_adam;

    DATA_TYPE* weighted_sum_output;
    DATA_TYPE* weighted_sum_grads;
} Attention_Layer;

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
        Attention_Layer attention_layer;
    } layer;
} Layer;

typedef struct NN {
    int num_layers;
    Layer* layers;
    int adamw_timestep;
    DATA_TYPE* pos_encoding;
    DATA_TYPE* pos_encoding_grads;
    AdamW_State pos_encoding_adam;

    int attn_residual_layer_idx;
    int ffn_residual_layer_idx;
    DATA_TYPE grad_scale;
} NN;

#endif
