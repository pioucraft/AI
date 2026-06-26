#include <cuda_runtime.h>
#include <stdio.h>

#include "convolution.h"
#include "dropout.h"
#include "mlp.h"
#include "nn.h"
#include "pooling.h"
#include "relu.h"
#include "softmax.h"
#include "tanh.h"
#include "utils.h"
#include "layernorm.h"
#include "attention.h"

int create_nn(NN* nn) {
    DATA_TYPE* current_input = NULL;
    DATA_TYPE* current_input_grads = NULL;

    for(int i = 0; i < nn->num_layers; i++) {
        Layer* layer = &(nn->layers[i]);
        if(layer->layer_type == LAYER_TYPE_MLP || layer->layer_type == LAYER_TYPE_LAYERNORM || layer->layer_type == LAYER_TYPE_SOFTMAX || layer->layer_type == LAYER_TYPE_ATTENTION) {
            layer->input.tensor.input = current_input;
            layer->input.tensor.grads = current_input_grads;

            int tensor_output_size = layer->output.tensor.output_size;

            if(i == nn->num_layers - 1) {
                cudaMallocManaged(&(current_input), layer->num_out_channels * tensor_output_size * sizeof(DATA_TYPE));
            } else {
                cudaMalloc(&(current_input), layer->num_out_channels * tensor_output_size * sizeof(DATA_TYPE));
            }

            cudaMalloc(&(current_input_grads), layer->num_out_channels * tensor_output_size * sizeof(DATA_TYPE));

            layer->output.tensor.output = current_input;
            layer->output.tensor.grads = current_input_grads;
        } else if(layer->layer_type == LAYER_TYPE_RELU || layer->layer_type == LAYER_TYPE_TANH || layer->layer_type == LAYER_TYPE_DROPOUT) { // 1d input and 1d output
            layer->input.d1.input = current_input;
            layer->input.d1.grads = current_input_grads;

            if(i == nn->num_layers - 1) {
                cudaMallocManaged(&(current_input), layer->num_out_channels * layer->output.d1.output_size * sizeof(DATA_TYPE));
            } else {
                cudaMalloc(&(current_input), layer->num_out_channels * layer->output.d1.output_size * sizeof(DATA_TYPE));
            }

            cudaMalloc(&(current_input_grads), layer->num_out_channels * layer->output.d1.output_size * sizeof(DATA_TYPE));

            layer->output.d1.output = current_input;
            layer->output.d1.grads = current_input_grads;
        } else if(layer->layer_type == LAYER_TYPE_POOLING || layer->layer_type == LAYER_TYPE_CONVOLUTION) { // 2d input and 2d output
            layer->input.d2.input = current_input;
            layer->input.d2.grads = current_input_grads;

            cudaMalloc(&(current_input), layer->num_out_channels * layer->output.d2.output_dimensions * layer->output.d2.output_dimensions * sizeof(DATA_TYPE));
            cudaMalloc(&(current_input_grads), layer->num_out_channels * layer->output.d2.output_dimensions * layer->output.d2.output_dimensions * sizeof(DATA_TYPE));

            layer->output.d2.output = current_input;
            layer->output.d2.grads = current_input_grads;
        }
    }

    checkCudaError();

    return 0;
}

int call_nn(NN* nn, DATA_TYPE* input, int run_dropout) {
    if(nn->layers[0].layer_type == LAYER_TYPE_MLP || nn->layers[0].layer_type == LAYER_TYPE_LAYERNORM || nn->layers[0].layer_type == LAYER_TYPE_SOFTMAX || nn->layers[0].layer_type == LAYER_TYPE_ATTENTION) {
        nn->layers[0].input.tensor.input = input;
    } else if(nn->layers[0].layer_type == LAYER_TYPE_RELU || nn->layers[0].layer_type == LAYER_TYPE_TANH || nn->layers[0].layer_type == LAYER_TYPE_DROPOUT) { // 1d input and 1d output
        nn->layers[0].input.d1.input = input;
    } else if(nn->layers[0].layer_type == LAYER_TYPE_POOLING || nn->layers[0].layer_type == LAYER_TYPE_CONVOLUTION) { // 2d input and 2d output
        nn->layers[0].input.d2.input = input;
    }

    for(int i = 0; i < nn->num_layers; i++) {
        Layer layer = nn->layers[i];
        if(layer.layer_type == LAYER_TYPE_MLP) {
            int batch_size = layer.input.tensor.tensor_dimensions[0];
            int input_feature_size = layer.input.tensor.tensor_dimensions[1];
            int output_feature_size = layer.output.tensor.tensor_dimensions[1];

            int num_blocks_bias = layer.output.tensor.output_size / NUM_THREADS + 1;
            bias_forward<<<num_blocks_bias, NUM_THREADS>>>(layer.output.tensor.output, layer.layer.mlp_layer.biases, layer.output.tensor.output_size, output_feature_size);
            cudaDeviceSynchronize();

            int num_blocks = batch_size * input_feature_size * output_feature_size / NUM_THREADS + 1;
            mlp_forward<<<num_blocks, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();

            if(i == 0 && nn->pos_encoding != NULL) {
                int total = layer.output.tensor.output_size;
                int nb = (total + NUM_THREADS - 1) / NUM_THREADS;
                add_pos_encoding<<<nb, NUM_THREADS>>>(layer.output.tensor.output, nn->pos_encoding, total);
                cudaDeviceSynchronize();
            }
        } else if(layer.layer_type == LAYER_TYPE_POOLING) {
            int num_blocks = layer.num_out_channels * layer.output.d2.output_dimensions * layer.output.d2.output_dimensions / NUM_THREADS + 1;
            pooling_forward<<<num_blocks, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();
        } else if(layer.layer_type == LAYER_TYPE_CONVOLUTION) {
            int num_blocks = layer.num_out_channels * layer.output.d2.output_dimensions * layer.output.d2.output_dimensions / NUM_THREADS + 1;
            convolution_forward<<<num_blocks, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();
        } else if(layer.layer_type == LAYER_TYPE_RELU) {
            int num_blocks = layer.output.d1.output_size / NUM_THREADS + 1;
            relu_forward<<<num_blocks, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();
        } else if(layer.layer_type == LAYER_TYPE_TANH) {
            int num_blocks = layer.output.d1.output_size / NUM_THREADS + 1;
            tanh_forward<<<num_blocks, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();
        } else if(layer.layer_type == LAYER_TYPE_DROPOUT) {
            int num_blocks = layer.output.d1.output_size / NUM_THREADS + 1;
            dropout_forward<<<num_blocks, NUM_THREADS>>>(layer, run_dropout);
            cudaDeviceSynchronize();
        } else if(layer.layer_type == LAYER_TYPE_LAYERNORM) {

            int num_inputs = 1;
            for(int j = 0; j < layer.input.tensor.tensor_rank; j++) {
                num_inputs *= layer.input.tensor.tensor_dimensions[j];
            }
            int num_blocks = num_inputs / NUM_THREADS + 1;
            layernorm_forward_zero_variance_mean<<<num_blocks, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();
            layernorm_forward_mean<<<num_blocks, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();
            layernorm_forward_variance<<<num_blocks, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();
            layernorm_forward<<<num_blocks, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();
        } else if(layer.layer_type == LAYER_TYPE_SOFTMAX) {
            int num_blocks_zero_exp_sums = layer.output.tensor.tensor_dimensions[0] / NUM_THREADS + 1;
            softmax_zero_exp_sums<<<num_blocks_zero_exp_sums, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();

            int num_blocks = layer.output.tensor.output_size / NUM_THREADS + 1;
            softmax_compute_exps<<<num_blocks, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();
            softmax_compute_outputs<<<num_blocks, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();
        } else if(layer.layer_type == LAYER_TYPE_ATTENTION) {
            int qk_size = layer.layer.attention_layer.context_length * layer.layer.attention_layer.query_key_size * layer.layer.attention_layer.num_heads;
            int v_size = layer.layer.attention_layer.context_length * layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.num_heads;
            int scores_size = layer.layer.attention_layer.context_length * layer.layer.attention_layer.context_length * layer.layer.attention_layer.num_heads;
            int sums_size = layer.layer.attention_layer.context_length * layer.layer.attention_layer.num_heads;
            int output_size = layer.layer.attention_layer.context_length * layer.layer.attention_layer.embedding_size;

            int num_blocks_qk = qk_size / NUM_THREADS + 1;
            attention_forward_key_query<<<num_blocks_qk, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();

            int num_blocks_v = v_size / NUM_THREADS + 1;
            attention_forward_value<<<num_blocks_v, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();

            int num_blocks_scores = scores_size / NUM_THREADS + 1;
            attention_forward_scores<<<num_blocks_scores, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();

            attention_forward_masking<<<num_blocks_scores, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();

            int num_blocks_sums = sums_size / NUM_THREADS + 1;
            attention_softmax_zero_exp_sums<<<num_blocks_sums, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();

            attention_softmax_compute_exps<<<num_blocks_scores, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();

            attention_softmax_compute_outputs<<<num_blocks_scores, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();

            int num_blocks_output = output_size / NUM_THREADS + 1;
            attention_forward_value_weighted_sum<<<num_blocks_output, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();
        }
    }

    checkCudaError();

    return 0;
}

__global__ void zero_grads_layer_1d_output(Layer layer) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(output_idx < layer.output.d1.output_size) {
        layer.output.d1.grads[output_idx] = (DATA_TYPE)0.0;
    }
}

__global__ void zero_grads_layer_2d_output(Layer layer) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(output_idx < layer.output.d2.output_dimensions * layer.output.d2.output_dimensions * layer.num_out_channels) {
        layer.output.d2.grads[output_idx] = (DATA_TYPE)0.0;
    }
}

__global__ void zero_grads_layer_tensor_output(Layer layer) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int output_size = 1;
    for(int i = 0; i < layer.output.tensor.tensor_rank; i++) {
        output_size *= layer.output.tensor.tensor_dimensions[i];
    }
    output_size *= layer.num_out_channels;

    if(output_idx < output_size) {
        layer.output.tensor.grads[output_idx] = (DATA_TYPE)0.0;
    }
}

int zero_grads_nn(NN* nn) {
    for(int i = 0; i < nn->num_layers; i++) {
        Layer layer = nn->layers[i];
        if(layer.layer_type == LAYER_TYPE_MLP || layer.layer_type == LAYER_TYPE_LAYERNORM || layer.layer_type == LAYER_TYPE_SOFTMAX || layer.layer_type == LAYER_TYPE_ATTENTION) { // tensor input and tensor output
            int output_size = layer.output.tensor.output_size;
            int num_blocks = output_size * layer.num_out_channels / NUM_THREADS + 1;
            zero_grads_layer_tensor_output<<<num_blocks, NUM_THREADS>>>(layer);
        } else if(layer.layer_type == LAYER_TYPE_RELU || layer.layer_type == LAYER_TYPE_TANH || layer.layer_type == LAYER_TYPE_DROPOUT) { // 1d input and 1d output
            int num_blocks = layer.output.d1.output_size * layer.num_out_channels / NUM_THREADS + 1;
            zero_grads_layer_1d_output<<<num_blocks, NUM_THREADS>>>(layer);
        } else if(layer.layer_type == LAYER_TYPE_POOLING || layer.layer_type == LAYER_TYPE_CONVOLUTION) { // 2d input and 2d output
            int num_blocks = layer.output.d2.output_dimensions * layer.output.d2.output_dimensions * layer.num_out_channels / NUM_THREADS + 1;
            zero_grads_layer_2d_output<<<num_blocks, NUM_THREADS>>>(layer);
        }

        if(layer.layer_type == LAYER_TYPE_MLP) {
            int input_feature_size = layer.input.tensor.tensor_dimensions[1];
            int output_feature_size = layer.output.tensor.tensor_dimensions[1];
            int num_blocks = input_feature_size * output_feature_size / NUM_THREADS + 1;
            zero_grads_mlp_layer<<<num_blocks, NUM_THREADS>>>(layer);
        } else if(layer.layer_type == LAYER_TYPE_CONVOLUTION) {
            int num_blocks = layer.layer.convolution_layer.filters_num * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions / NUM_THREADS + 1;
            zero_grads_convolution_layer<<<num_blocks, NUM_THREADS>>>(layer);
        } else if(layer.layer_type == LAYER_TYPE_LAYERNORM) {
            int num_blocks = layer.input.tensor.tensor_dimensions[0] * layer.num_out_channels / NUM_THREADS + 1;
            zero_grads_layernorm_layer<<<num_blocks, NUM_THREADS>>>(layer);
        } else if(layer.layer_type == LAYER_TYPE_ATTENTION) {
            int qk_weight_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.query_key_size * layer.layer.attention_layer.num_heads;
            int v_weight_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.num_heads;
            int num_blocks = (qk_weight_size + qk_weight_size + v_weight_size) / NUM_THREADS + 1;
            zero_grads_attention_layer<<<num_blocks, NUM_THREADS>>>(layer);
        }
    }

    if(nn->pos_encoding != NULL) {
        cudaMemset(nn->pos_encoding_grads, 0, nn->layers[0].output.tensor.output_size * sizeof(DATA_TYPE));
    }

    cudaDeviceSynchronize();
    checkCudaError();

    return 0;
}

__global__ void grad_error(Layer output_layer, DATA_TYPE* expected_output) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(output_layer.layer_type == LAYER_TYPE_TANH) {
        if(output_idx >= output_layer.output.d1.output_size) return;
        DATA_TYPE error_grad = 2 * (output_layer.output.d1.output[output_idx] - expected_output[output_idx]);
        output_layer.output.d1.grads[output_idx] = error_grad;
    }
}

int grad_nn(NN* nn, DATA_TYPE* expected_output) {
    for(int i = nn->num_layers - 1; i >= 0; i--) {
        Layer layer = nn->layers[i];
        if(i == nn->num_layers - 1) {
            if(nn->layers[i].layer_type == LAYER_TYPE_TANH) {
                int output_size = layer.output.d1.output_size;
                int num_blocks = output_size / NUM_THREADS + 1;
                grad_error<<<num_blocks, NUM_THREADS>>>(layer, expected_output);
            }
        }
        cudaDeviceSynchronize();

        if(layer.layer_type == LAYER_TYPE_MLP) {
            if(layer.input.tensor.grads != NULL) {
                int num_blocks = layer.input.tensor.input_size / NUM_THREADS + 1;
                zero_input_grads_mlp_layer<<<num_blocks, NUM_THREADS>>>(layer);
                cudaDeviceSynchronize();
            }

            int batch_size = layer.input.tensor.tensor_dimensions[0];
            int input_feature_size = layer.input.tensor.tensor_dimensions[1];
            int output_feature_size = layer.output.tensor.tensor_dimensions[1];
            int num_blocks = batch_size * input_feature_size * output_feature_size / NUM_THREADS + 1;
            grad_mlp_layer<<<num_blocks, NUM_THREADS>>>(layer);
        } else if(layer.layer_type == LAYER_TYPE_POOLING) {
            if(layer.input.d2.grads != NULL) {
                int num_blocks = layer.num_in_channels * layer.input.d2.input_dimensions * layer.input.d2.input_dimensions / NUM_THREADS + 1;
                zero_input_grads_pooling_layer<<<num_blocks, NUM_THREADS>>>(layer);
                cudaDeviceSynchronize();
            }
            int num_blocks = layer.num_out_channels * layer.output.d2.output_dimensions * layer.output.d2.output_dimensions / NUM_THREADS + 1;
            grad_pooling_layer<<<num_blocks, NUM_THREADS>>>(layer);
        } else if(layer.layer_type == LAYER_TYPE_CONVOLUTION) {
            if(layer.input.d2.grads != NULL) {
                int num_blocks = layer.num_in_channels * layer.input.d2.input_dimensions * layer.input.d2.input_dimensions / NUM_THREADS + 1;
                zero_input_grads_convolution_layer<<<num_blocks, NUM_THREADS>>>(layer);
                cudaDeviceSynchronize();
            }

            int num_blocks = layer.num_out_channels * layer.output.d2.output_dimensions * layer.output.d2.output_dimensions / NUM_THREADS + 1;
            grad_convolution_layer<<<num_blocks, NUM_THREADS>>>(layer);
        } else if(layer.layer_type == LAYER_TYPE_RELU) {
            if(layer.input.d1.grads != NULL) {
                int num_blocks = layer.input.d1.input_size / NUM_THREADS + 1;
                zero_input_grads_relu_layer<<<num_blocks, NUM_THREADS>>>(layer);
                cudaDeviceSynchronize();
            }
            int num_blocks = layer.output.d1.output_size / NUM_THREADS + 1;
            grad_relu_layer<<<num_blocks, NUM_THREADS>>>(layer);
        } else if(layer.layer_type == LAYER_TYPE_TANH) {
            if(layer.input.d1.grads != NULL) {
                int num_blocks = layer.input.d1.input_size / NUM_THREADS + 1;
                zero_input_grads_tanh_layer<<<num_blocks, NUM_THREADS>>>(layer);
                cudaDeviceSynchronize();
            }
            int num_blocks = layer.output.d1.output_size / NUM_THREADS + 1;
            grad_tanh_layer<<<num_blocks, NUM_THREADS>>>(layer);
        } else if(layer.layer_type == LAYER_TYPE_DROPOUT) {
            if(layer.input.d1.grads != NULL) {
                int num_blocks = layer.input.d1.input_size / NUM_THREADS + 1;
                zero_input_grads_dropout_layer<<<num_blocks, NUM_THREADS>>>(layer);
                cudaDeviceSynchronize();
            }
            int num_blocks = layer.output.d1.output_size / NUM_THREADS + 1;
            grad_dropout_layer<<<num_blocks, NUM_THREADS>>>(layer);
        } else if(layer.layer_type == LAYER_TYPE_LAYERNORM) {
            if(layer.input.tensor.grads != NULL) {
                int num_blocks = layer.input.tensor.input_size / NUM_THREADS + 1;
                zero_input_grads_layernorm_layer<<<num_blocks, NUM_THREADS>>>(layer);
                cudaDeviceSynchronize();
            }
            int num_blocks = layer.output.tensor.output_size / NUM_THREADS + 1;
            grad_layernorm_layer<<<num_blocks, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();
            grad_layernorm_layer_step_two<<<num_blocks, NUM_THREADS>>>(layer);
        } else if(layer.layer_type == LAYER_TYPE_SOFTMAX) {
            if(i == nn->num_layers - 1) {
                int num_blocks = layer.input.tensor.input_size / NUM_THREADS + 1;
                grad_softmax_simplified<<<num_blocks, NUM_THREADS>>>(layer, expected_output);
            } else {
                if(layer.input.tensor.grads != NULL) {
                    int num_blocks = layer.input.tensor.input_size / NUM_THREADS + 1;
                    zero_input_grads_softmax_layer<<<num_blocks, NUM_THREADS>>>(layer);
                    cudaDeviceSynchronize();
                }
                int num_blocks = layer.output.tensor.output_size / NUM_THREADS + 1;
                grad_softmax_layer_step_1<<<num_blocks, NUM_THREADS>>>(layer);
                cudaDeviceSynchronize();
                grad_softmax_layer_step_2<<<num_blocks, NUM_THREADS>>>(layer);
            }
        } else if(layer.layer_type == LAYER_TYPE_ATTENTION) {
            int qk_size = layer.layer.attention_layer.context_length * layer.layer.attention_layer.query_key_size * layer.layer.attention_layer.num_heads;
            int v_size = layer.layer.attention_layer.context_length * layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.num_heads;
            int scores_size = layer.layer.attention_layer.context_length * layer.layer.attention_layer.context_length * layer.layer.attention_layer.num_heads;
            int sums_size = layer.layer.attention_layer.context_length * layer.layer.attention_layer.num_heads;
            int output_size = layer.layer.attention_layer.context_length * layer.layer.attention_layer.embedding_size;

            if(layer.input.tensor.grads != NULL) {
                int input_size = layer.input.tensor.input_size;
                int total = input_size + qk_size + v_size + scores_size + scores_size + scores_size + sums_size;
                int num_blocks = total / NUM_THREADS + 1;
                zero_input_grads_attention_layer<<<num_blocks, NUM_THREADS>>>(layer);
                cudaDeviceSynchronize();
            }

            int num_blocks_output = output_size / NUM_THREADS + 1;
            grad_attention_layer_value_weighted_sum<<<num_blocks_output, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();

            int num_blocks_scores = scores_size / NUM_THREADS + 1;
            grad_attention_layer_softmax_step_1<<<num_blocks_scores, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();
            grad_attention_layer_softmax_step_2<<<num_blocks_scores, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();
            grad_attention_layer_masking<<<num_blocks_scores, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();
            grad_attention_layer_scores<<<num_blocks_scores, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();

            int num_blocks_qk = qk_size / NUM_THREADS + 1;
            grad_attention_layer_key_query<<<num_blocks_qk, NUM_THREADS>>>(layer);
            cudaDeviceSynchronize();

            int num_blocks_v = v_size / NUM_THREADS + 1;
            grad_attention_layer_value<<<num_blocks_v, NUM_THREADS>>>(layer);
        }

        cudaDeviceSynchronize();
    }

    if(nn->pos_encoding != NULL) {
        int total_size = nn->layers[0].output.tensor.output_size;
        int nb = (total_size + NUM_THREADS - 1) / NUM_THREADS;
        grad_pos_encoding<<<nb, NUM_THREADS>>>(nn->pos_encoding_grads, nn->layers[0].output.tensor.grads, total_size);
    }

    checkCudaError();

    return 0;
}


int update_nn(NN* nn, DATA_TYPE learning_rate, DATA_TYPE weight_decay) {
    nn->adamw_timestep++;

    for(int i = 0; i < nn->num_layers; i++) {
        Layer layer = nn->layers[i];
        if(layer.layer_type == LAYER_TYPE_MLP) {
            int input_feature_size = layer.input.tensor.tensor_dimensions[1];
            int output_feature_size = layer.output.tensor.tensor_dimensions[1];
            int num_blocks = input_feature_size * output_feature_size / NUM_THREADS + 1;
            update_mlp_layer<<<num_blocks, NUM_THREADS>>>(layer, learning_rate, nn->adamw_timestep, weight_decay);
        } else if(layer.layer_type == LAYER_TYPE_CONVOLUTION) {
            int num_blocks = layer.layer.convolution_layer.filters_num * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions / NUM_THREADS + 1;
            update_convolution_layer<<<num_blocks, NUM_THREADS>>>(layer, learning_rate, nn->adamw_timestep, weight_decay);
        } else if(layer.layer_type == LAYER_TYPE_LAYERNORM) {
            int num_blocks = layer.input.tensor.tensor_dimensions[0] * layer.num_out_channels / NUM_THREADS + 1;
            update_layernorm_layer<<<num_blocks, NUM_THREADS>>>(layer, learning_rate, nn->adamw_timestep, weight_decay);
        } else if(layer.layer_type == LAYER_TYPE_ATTENTION) {
            int qk_weight_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.query_key_size * layer.layer.attention_layer.num_heads;
            int v_weight_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.num_heads;
            int num_blocks = (qk_weight_size + qk_weight_size + v_weight_size) / NUM_THREADS + 1;
            update_attention_layer<<<num_blocks, NUM_THREADS>>>(layer, learning_rate, nn->adamw_timestep, weight_decay);
        }
    }

    if(nn->pos_encoding != NULL) {
        int total_size = nn->layers[0].output.tensor.output_size;
        int nb = (total_size + NUM_THREADS - 1) / NUM_THREADS;
        update_pos_encoding<<<nb, NUM_THREADS>>>(nn->pos_encoding, nn->pos_encoding_grads,
            nn->pos_encoding_adam, total_size, learning_rate, nn->adamw_timestep, weight_decay);
    }

    cudaDeviceSynchronize();
    checkCudaError();

    return 0;
}

int save_nn(NN* nn, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if(file == NULL) {
        printf("Error opening file for writing: %s\n", filename);
        return -1;
    }

    for(int i = 0; i < nn->num_layers; i++) {
        Layer layer = nn->layers[i];
        if(layer.layer_type == LAYER_TYPE_MLP) {
            save_mlp_layer(layer, file);
        } else if(layer.layer_type == LAYER_TYPE_CONVOLUTION) {
            save_convolution_layer(layer, file);
        } else if(layer.layer_type == LAYER_TYPE_LAYERNORM) {
            save_layernorm_layer(layer, file);
        } else if(layer.layer_type == LAYER_TYPE_ATTENTION) {
            save_attention_layer(layer, file);
        }
    }

    if(nn->pos_encoding != NULL) {
        int total_size = nn->layers[0].output.tensor.output_size;
        DATA_TYPE* host_pos = (DATA_TYPE*)malloc(total_size * sizeof(DATA_TYPE));
        cudaMemcpy(host_pos, nn->pos_encoding, total_size * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
        fwrite(host_pos, sizeof(DATA_TYPE), total_size, file);
        free(host_pos);
    }

    fclose(file);
    return 0;
}

int load_nn(NN* nn, const char* filename) {
    FILE* file = fopen(filename, "rb");
    if(file == NULL) {
        printf("Error opening file for reading: %s\n", filename);
        return -1;
    }

    for(int i = 0; i < nn->num_layers; i++) {
        Layer* layer = &(nn->layers[i]);
        if(layer->layer_type == LAYER_TYPE_MLP) {
            load_mlp_layer(layer, file);
        } else if(layer->layer_type == LAYER_TYPE_CONVOLUTION) {
            load_convolution_layer(layer, file);
        } else if(layer->layer_type == LAYER_TYPE_LAYERNORM) {
            load_layernorm_layer(layer, file);
        } else if(layer->layer_type == LAYER_TYPE_ATTENTION) {
            load_attention_layer(layer, file);
        }
    }

    if(nn->pos_encoding != NULL) {
        int total_size = nn->layers[0].output.tensor.output_size;
        DATA_TYPE* host_pos = (DATA_TYPE*)malloc(total_size * sizeof(DATA_TYPE));
        fread(host_pos, sizeof(DATA_TYPE), total_size, file);
        cudaMemcpy(nn->pos_encoding, host_pos, total_size * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
        free(host_pos);
    }

    fclose(file);
    return 0;
}

__global__ void grad_sq_sum(DATA_TYPE* grads, int size, DATA_TYPE* result) {
    __shared__ DATA_TYPE shared[NUM_THREADS];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    shared[tid] = 0.0f;
    while(idx < size) {
        shared[tid] += grads[idx] * grads[idx];
        idx += gridDim.x * blockDim.x;
    }
    __syncthreads();

    for(int s = NUM_THREADS / 2; s > 0; s >>= 1) {
        if(tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    if(tid == 0) atomicAdd(result, shared[0]);
}

__global__ void scale_grads(DATA_TYPE* grads, int size, DATA_TYPE scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while(idx < size) {
        grads[idx] *= scale;
        idx += gridDim.x * blockDim.x;
    }
}

int clip_grads_nn(NN* nn, DATA_TYPE max_norm) {
    DATA_TYPE* d_sq_sum;
    cudaMalloc(&d_sq_sum, sizeof(DATA_TYPE));
    cudaMemset(d_sq_sum, 0, sizeof(DATA_TYPE));

    for(int i = 0; i < nn->num_layers; i++) {
        Layer layer = nn->layers[i];

        if(layer.layer_type == LAYER_TYPE_MLP) {
            int input_feature_size = layer.input.tensor.tensor_dimensions[1];
            int output_feature_size = layer.output.tensor.tensor_dimensions[1];
            int weight_size = input_feature_size * output_feature_size;

            if(layer.layer.mlp_layer.weight_grads != NULL) {
                int num_blocks = weight_size / NUM_THREADS + 1;
                grad_sq_sum<<<num_blocks, NUM_THREADS>>>(layer.layer.mlp_layer.weight_grads, weight_size, d_sq_sum);
            }
            if(layer.layer.mlp_layer.bias_grads != NULL) {
                int num_blocks = output_feature_size / NUM_THREADS + 1;
                grad_sq_sum<<<num_blocks, NUM_THREADS>>>(layer.layer.mlp_layer.bias_grads, output_feature_size, d_sq_sum);
            }
        } else if(layer.layer_type == LAYER_TYPE_CONVOLUTION) {
            int filter_size = layer.layer.convolution_layer.filters_num * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions * layer.num_in_channels;

            if(layer.layer.convolution_layer.filter_grads != NULL) {
                int num_blocks = filter_size / NUM_THREADS + 1;
                grad_sq_sum<<<num_blocks, NUM_THREADS>>>(layer.layer.convolution_layer.filter_grads, filter_size, d_sq_sum);
            }
            if(layer.layer.convolution_layer.bias_grads != NULL) {
                int num_blocks = layer.layer.convolution_layer.filters_num / NUM_THREADS + 1;
                grad_sq_sum<<<num_blocks, NUM_THREADS>>>(layer.layer.convolution_layer.bias_grads, layer.layer.convolution_layer.filters_num, d_sq_sum);
            }
        } else if(layer.layer_type == LAYER_TYPE_LAYERNORM) {
            int vector_size = layer.input.tensor.tensor_dimensions[1];

            if(layer.layer.layernorm_layer.gain_grads != NULL) {
                int num_blocks = vector_size / NUM_THREADS + 1;
                grad_sq_sum<<<num_blocks, NUM_THREADS>>>(layer.layer.layernorm_layer.gain_grads, vector_size, d_sq_sum);
            }
            if(layer.layer.layernorm_layer.bias_grads != NULL) {
                int num_blocks = vector_size / NUM_THREADS + 1;
                grad_sq_sum<<<num_blocks, NUM_THREADS>>>(layer.layer.layernorm_layer.bias_grads, vector_size, d_sq_sum);
            }
        } else if(layer.layer_type == LAYER_TYPE_ATTENTION) {
            int qk_weight_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.query_key_size * layer.layer.attention_layer.num_heads;
            int v_weight_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.num_heads;

            if(layer.layer.attention_layer.query_weight_grads != NULL) {
                int num_blocks = qk_weight_size / NUM_THREADS + 1;
                grad_sq_sum<<<num_blocks, NUM_THREADS>>>(layer.layer.attention_layer.query_weight_grads, qk_weight_size, d_sq_sum);
            }
            if(layer.layer.attention_layer.key_weight_grads != NULL) {
                int num_blocks = qk_weight_size / NUM_THREADS + 1;
                grad_sq_sum<<<num_blocks, NUM_THREADS>>>(layer.layer.attention_layer.key_weight_grads, qk_weight_size, d_sq_sum);
            }
            if(layer.layer.attention_layer.value_weight_grads != NULL) {
                int num_blocks = v_weight_size / NUM_THREADS + 1;
                grad_sq_sum<<<num_blocks, NUM_THREADS>>>(layer.layer.attention_layer.value_weight_grads, v_weight_size, d_sq_sum);
            }
        }
    }

    if(nn->pos_encoding != NULL) {
        int total_size = nn->layers[0].output.tensor.output_size;
        int num_blocks = total_size / NUM_THREADS + 1;
        grad_sq_sum<<<num_blocks, NUM_THREADS>>>(nn->pos_encoding_grads, total_size, d_sq_sum);
    }

    cudaDeviceSynchronize();

    DATA_TYPE h_sq_sum;
    cudaMemcpy(&h_sq_sum, d_sq_sum, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaFree(d_sq_sum);

    DATA_TYPE norm = sqrtf(h_sq_sum);
    if(norm > max_norm) {
        DATA_TYPE scale = max_norm / norm;

        for(int i = 0; i < nn->num_layers; i++) {
            Layer layer = nn->layers[i];

            if(layer.layer_type == LAYER_TYPE_MLP) {
                int input_feature_size = layer.input.tensor.tensor_dimensions[1];
                int output_feature_size = layer.output.tensor.tensor_dimensions[1];
                int weight_size = input_feature_size * output_feature_size;

                if(layer.layer.mlp_layer.weight_grads != NULL) {
                    int num_blocks = weight_size / NUM_THREADS + 1;
                    scale_grads<<<num_blocks, NUM_THREADS>>>(layer.layer.mlp_layer.weight_grads, weight_size, scale);
                }
                if(layer.layer.mlp_layer.bias_grads != NULL) {
                    int num_blocks = output_feature_size / NUM_THREADS + 1;
                    scale_grads<<<num_blocks, NUM_THREADS>>>(layer.layer.mlp_layer.bias_grads, output_feature_size, scale);
                }
            } else if(layer.layer_type == LAYER_TYPE_CONVOLUTION) {
                int filter_size = layer.layer.convolution_layer.filters_num * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions * layer.num_in_channels;

                if(layer.layer.convolution_layer.filter_grads != NULL) {
                    int num_blocks = filter_size / NUM_THREADS + 1;
                    scale_grads<<<num_blocks, NUM_THREADS>>>(layer.layer.convolution_layer.filter_grads, filter_size, scale);
                }
                if(layer.layer.convolution_layer.bias_grads != NULL) {
                    int num_blocks = layer.layer.convolution_layer.filters_num / NUM_THREADS + 1;
                    scale_grads<<<num_blocks, NUM_THREADS>>>(layer.layer.convolution_layer.bias_grads, layer.layer.convolution_layer.filters_num, scale);
                }
            } else if(layer.layer_type == LAYER_TYPE_LAYERNORM) {
                int vector_size = layer.input.tensor.tensor_dimensions[1];

                if(layer.layer.layernorm_layer.gain_grads != NULL) {
                    int num_blocks = vector_size / NUM_THREADS + 1;
                    scale_grads<<<num_blocks, NUM_THREADS>>>(layer.layer.layernorm_layer.gain_grads, vector_size, scale);
                }
                if(layer.layer.layernorm_layer.bias_grads != NULL) {
                    int num_blocks = vector_size / NUM_THREADS + 1;
                    scale_grads<<<num_blocks, NUM_THREADS>>>(layer.layer.layernorm_layer.bias_grads, vector_size, scale);
                }
            } else if(layer.layer_type == LAYER_TYPE_ATTENTION) {
                int qk_weight_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.query_key_size * layer.layer.attention_layer.num_heads;
                int v_weight_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.num_heads;

                if(layer.layer.attention_layer.query_weight_grads != NULL) {
                    int num_blocks = qk_weight_size / NUM_THREADS + 1;
                    scale_grads<<<num_blocks, NUM_THREADS>>>(layer.layer.attention_layer.query_weight_grads, qk_weight_size, scale);
                }
                if(layer.layer.attention_layer.key_weight_grads != NULL) {
                    int num_blocks = qk_weight_size / NUM_THREADS + 1;
                    scale_grads<<<num_blocks, NUM_THREADS>>>(layer.layer.attention_layer.key_weight_grads, qk_weight_size, scale);
                }
                if(layer.layer.attention_layer.value_weight_grads != NULL) {
                    int num_blocks = v_weight_size / NUM_THREADS + 1;
                    scale_grads<<<num_blocks, NUM_THREADS>>>(layer.layer.attention_layer.value_weight_grads, v_weight_size, scale);
                }
            }
        }

        if(nn->pos_encoding != NULL) {
            int total_size = nn->layers[0].output.tensor.output_size;
            int num_blocks = total_size / NUM_THREADS + 1;
            scale_grads<<<num_blocks, NUM_THREADS>>>(nn->pos_encoding_grads, total_size, scale);
        }
    }

    cudaDeviceSynchronize();
    checkCudaError();

    return 0;
}

__global__ void add_pos_encoding(DATA_TYPE* output, DATA_TYPE* pos_encoding, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total_size) {
        output[idx] += pos_encoding[idx];
    }
}

__global__ void grad_pos_encoding(DATA_TYPE* pos_grads, DATA_TYPE* grads, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total_size) {
        atomicAdd(&pos_grads[idx], grads[idx]);
    }
}

__global__ void update_pos_encoding(DATA_TYPE* pos, DATA_TYPE* pos_grads, AdamW_State adam,
                                    int total_size, DATA_TYPE lr, int timestep, DATA_TYPE wd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= total_size) return;

    DATA_TYPE grad = pos_grads[idx];
    DATA_TYPE weight = pos[idx];
    DATA_TYPE m = adam.m[idx];
    DATA_TYPE v = adam.v[idx];

    m = ADAMW_BETA1 * m + (1.0f - ADAMW_BETA1) * grad;
    v = ADAMW_BETA2 * v + (1.0f - ADAMW_BETA2) * grad * grad;

    DATA_TYPE m_hat = m / (1.0f - powf(ADAMW_BETA1, timestep));
    DATA_TYPE v_hat = v / (1.0f - powf(ADAMW_BETA2, timestep));

    adam.m[idx] = m;
    adam.v[idx] = v;

    pos[idx] = weight - lr * m_hat / (sqrtf(v_hat) + ADAMW_EPSILON) - lr * wd * weight;
}
