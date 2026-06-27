#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <stdio.h>

#include "mlp.h"
#include "nn.h"
#include "utils.h"

__global__ void bias_forward(DATA_TYPE* outputs, DATA_TYPE* biases, int total_elements, int output_feature_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < total_elements) {
        outputs[idx] = biases[idx % output_feature_size];
    }
}

int create_mlp_layer(Layer* layer, int tensor_rank, int tensor_dimensions[TENSOR_MAX_RANK], int output_feature_size) {
    int batch_size = tensor_dimensions[0];
    int input_feature_size = tensor_dimensions[1];

    int input_size = batch_size * input_feature_size;
    int output_size = batch_size * output_feature_size;

    DATA_TYPE* weights;
    DATA_TYPE* biases;

    cudaMalloc(&weights, input_feature_size * output_feature_size * sizeof(DATA_TYPE));
    cudaMalloc(&biases, output_feature_size * sizeof(DATA_TYPE));

    for(int i = 0; i < input_feature_size * output_feature_size; i++) {
        DATA_TYPE weight = normal_sample(0.0f, 0.02f);
        cudaMemcpy(weights + i, &weight, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    }

    for(int i = 0; i < output_feature_size; i++) {
        DATA_TYPE bias = 0.0f;
        cudaMemcpy(biases + i, &bias, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    }

    DATA_TYPE* weight_grads;
    DATA_TYPE* bias_grads;

    cudaMalloc(&weight_grads, input_feature_size * output_feature_size * sizeof(DATA_TYPE));
    cudaMalloc(&bias_grads, output_feature_size * sizeof(DATA_TYPE));

    AdamW_State weights_adam, biases_adam;
    cudaMalloc(&weights_adam.m, input_feature_size * output_feature_size * sizeof(DATA_TYPE));
    cudaMalloc(&weights_adam.v, input_feature_size * output_feature_size * sizeof(DATA_TYPE));
    cudaMemset(weights_adam.m, 0, input_feature_size * output_feature_size * sizeof(DATA_TYPE));
    cudaMemset(weights_adam.v, 0, input_feature_size * output_feature_size * sizeof(DATA_TYPE));
    cudaMalloc(&biases_adam.m, output_feature_size * sizeof(DATA_TYPE));
    cudaMalloc(&biases_adam.v, output_feature_size * sizeof(DATA_TYPE));
    cudaMemset(biases_adam.m, 0, output_feature_size * sizeof(DATA_TYPE));
    cudaMemset(biases_adam.v, 0, output_feature_size * sizeof(DATA_TYPE));

    int output_tensor_dimensions[TENSOR_MAX_RANK] = {batch_size, output_feature_size, 0, 0};

    *layer = (Layer){
        .layer_type = LAYER_TYPE_MLP,
        .num_in_channels = 1,
        .num_out_channels = 1,
        .input = {
            .tensor = {
                .tensor_rank = 2,
                .input_size = input_size
            }
        },
        .output = {
            .tensor = {
                .tensor_rank = 2,
                .output_size = output_size
            }
        },
        .layer = {
            .mlp_layer = {
                .weights = weights,
                .biases = biases,

                .weight_grads = weight_grads,
                .bias_grads = bias_grads,

                .weights_adam = weights_adam,
                .biases_adam = biases_adam
            }
        }
    };
    layer->input.tensor.tensor_dimensions[0] = batch_size;
    layer->input.tensor.tensor_dimensions[1] = input_feature_size;
    layer->output.tensor.tensor_dimensions[0] = batch_size;
    layer->output.tensor.tensor_dimensions[1] = output_feature_size;

    return 0;
}

__global__ void mlp_forward(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int batch_size = layer.input.tensor.tensor_dimensions[0];
    int input_feature_size = layer.input.tensor.tensor_dimensions[1];
    int output_feature_size = layer.output.tensor.tensor_dimensions[1];

    int inner_size = input_feature_size * output_feature_size;

    if(idx >= batch_size * inner_size) {
        return;
    }

    int batch_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    int neuron_idx = inner_idx / input_feature_size;
    int input_idx = inner_idx % input_feature_size;

    atomicAdd(&(layer.output.tensor.output[batch_idx * output_feature_size + neuron_idx]),
              layer.input.tensor.input[batch_idx * input_feature_size + input_idx] * layer.layer.mlp_layer.weights[inner_idx]);
}


__global__ void zero_grads_mlp_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int input_feature_size = layer.input.tensor.tensor_dimensions[1];
    int output_feature_size = layer.output.tensor.tensor_dimensions[1];

    int total_weights = input_feature_size * output_feature_size;

    if(idx >= total_weights) {
        return;
    }

    int neuron_idx = idx / input_feature_size;
    int input_idx = idx % input_feature_size;

    if(input_idx == 0) {
        layer.layer.mlp_layer.bias_grads[neuron_idx] = (DATA_TYPE)0.0;
    }
    layer.layer.mlp_layer.weight_grads[idx] = (DATA_TYPE)0.0;
}

__global__ void zero_input_grads_mlp_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= layer.input.tensor.input_size) {
        return;
    }

    layer.input.tensor.grads[idx] = (DATA_TYPE)0.0;
}

__global__ void grad_mlp_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int batch_size = layer.input.tensor.tensor_dimensions[0];
    int input_feature_size = layer.input.tensor.tensor_dimensions[1];
    int output_feature_size = layer.output.tensor.tensor_dimensions[1];

    int inner_size = input_feature_size * output_feature_size;

    if(idx >= batch_size * inner_size) {
        return;
    }

    int batch_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    int neuron_idx = inner_idx / input_feature_size;
    int input_idx = inner_idx % input_feature_size;

    if(input_idx == 0) {
        atomicAdd(&(layer.layer.mlp_layer.bias_grads[neuron_idx]),
                  layer.output.tensor.grads[batch_idx * output_feature_size + neuron_idx]);
    }
    atomicAdd(&(layer.layer.mlp_layer.weight_grads[inner_idx]),
              layer.output.tensor.grads[batch_idx * output_feature_size + neuron_idx] * layer.input.tensor.input[batch_idx * input_feature_size + input_idx]);

    if(layer.input.tensor.grads != NULL) {
        atomicAdd(&(layer.input.tensor.grads[batch_idx * input_feature_size + input_idx]),
                  layer.output.tensor.grads[batch_idx * output_feature_size + neuron_idx] * layer.layer.mlp_layer.weights[inner_idx]);
    }
}

__global__ void update_mlp_layer(Layer layer, DATA_TYPE learning_rate, int timestep, DATA_TYPE weight_decay) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int input_feature_size = layer.input.tensor.tensor_dimensions[1];
    int output_feature_size = layer.output.tensor.tensor_dimensions[1];

    int total_weights = input_feature_size * output_feature_size;

    if(idx >= total_weights) {
        return;
    }

    int neuron_idx = idx / input_feature_size;
    int input_idx = idx % input_feature_size;

    if(input_idx == 0) {
        DATA_TYPE grad = layer.layer.mlp_layer.bias_grads[neuron_idx];
        DATA_TYPE bias = layer.layer.mlp_layer.biases[neuron_idx];
        DATA_TYPE m = layer.layer.mlp_layer.biases_adam.m[neuron_idx];
        DATA_TYPE v = layer.layer.mlp_layer.biases_adam.v[neuron_idx];

        m = ADAMW_BETA1 * m + (1.0f - ADAMW_BETA1) * grad;
        v = ADAMW_BETA2 * v + (1.0f - ADAMW_BETA2) * grad * grad;

        DATA_TYPE m_hat = m / (1.0f - powf(ADAMW_BETA1, timestep));
        DATA_TYPE v_hat = v / (1.0f - powf(ADAMW_BETA2, timestep));

        layer.layer.mlp_layer.biases_adam.m[neuron_idx] = m;
        layer.layer.mlp_layer.biases_adam.v[neuron_idx] = v;

        layer.layer.mlp_layer.biases[neuron_idx] = bias - learning_rate * m_hat / (sqrtf(v_hat) + ADAMW_EPSILON) - learning_rate * weight_decay * bias;
    }

    DATA_TYPE grad = layer.layer.mlp_layer.weight_grads[idx];
    DATA_TYPE weight = layer.layer.mlp_layer.weights[idx];
    DATA_TYPE m = layer.layer.mlp_layer.weights_adam.m[idx];
    DATA_TYPE v = layer.layer.mlp_layer.weights_adam.v[idx];

    m = ADAMW_BETA1 * m + (1.0f - ADAMW_BETA1) * grad;
    v = ADAMW_BETA2 * v + (1.0f - ADAMW_BETA2) * grad * grad;

    DATA_TYPE m_hat = m / (1.0f - powf(ADAMW_BETA1, timestep));
    DATA_TYPE v_hat = v / (1.0f - powf(ADAMW_BETA2, timestep));

    layer.layer.mlp_layer.weights_adam.m[idx] = m;
    layer.layer.mlp_layer.weights_adam.v[idx] = v;

    layer.layer.mlp_layer.weights[idx] = weight - learning_rate * m_hat / (sqrtf(v_hat) + ADAMW_EPSILON) - learning_rate * weight_decay * weight;
}

int save_mlp_layer(Layer layer, FILE* file) {
    int input_feature_size = layer.input.tensor.tensor_dimensions[1];
    int output_feature_size = layer.output.tensor.tensor_dimensions[1];

    DATA_TYPE* host_weights = (DATA_TYPE*)malloc(input_feature_size * output_feature_size * sizeof(DATA_TYPE));
    DATA_TYPE* host_biases = (DATA_TYPE*)malloc(output_feature_size * sizeof(DATA_TYPE));

    cudaMemcpy(host_weights, layer.layer.mlp_layer.weights, input_feature_size * output_feature_size * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_biases, layer.layer.mlp_layer.biases, output_feature_size * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

    fwrite(host_weights, sizeof(DATA_TYPE), input_feature_size * output_feature_size, file);
    fwrite(host_biases, sizeof(DATA_TYPE), output_feature_size, file);

    free(host_weights);
    free(host_biases);

    return 0;
}

int load_mlp_layer(Layer* layer, FILE* file) {
    int input_feature_size = layer->input.tensor.tensor_dimensions[1];
    int output_feature_size = layer->output.tensor.tensor_dimensions[1];

    DATA_TYPE* host_weights = (DATA_TYPE*)malloc(input_feature_size * output_feature_size * sizeof(DATA_TYPE));
    DATA_TYPE* host_biases = (DATA_TYPE*)malloc(output_feature_size * sizeof(DATA_TYPE));

    fread(host_weights, sizeof(DATA_TYPE), input_feature_size * output_feature_size, file);
    fread(host_biases, sizeof(DATA_TYPE), output_feature_size, file);

    cudaMemcpy(layer->layer.mlp_layer.weights, host_weights, input_feature_size * output_feature_size * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(layer->layer.mlp_layer.biases, host_biases, output_feature_size * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    free(host_weights);
    free(host_biases);

    return 0;
}
