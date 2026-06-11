#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <stdio.h>

#include "mlp.h"
#include "nn.h"
#include "utils.h"

int create_mlp_layer(Layer* layer, int input_size, int output_size) {
    DATA_TYPE* weights;
    DATA_TYPE* biases;

    cudaMalloc(&weights, input_size * output_size * sizeof(DATA_TYPE));
    cudaMalloc(&biases, output_size * sizeof(DATA_TYPE));

    DATA_TYPE deviation = sqrt(2.0 / (input_size));
    for(int i = 0; i < input_size * output_size; i++) {
        DATA_TYPE weight = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * deviation * 2 - deviation);
        cudaMemcpy(weights + i, &weight, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    }

    for(int i = 0; i < output_size; i++) {
        DATA_TYPE bias = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * deviation * 2 - deviation);
        cudaMemcpy(biases + i, &bias, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    }

    DATA_TYPE* weight_grads;
    DATA_TYPE* bias_grads;

    cudaMalloc(&weight_grads, input_size * output_size * sizeof(DATA_TYPE));
    cudaMalloc(&bias_grads, output_size * sizeof(DATA_TYPE));

    *layer = {
        .layer_type = LAYER_TYPE_MLP,
        .num_in_channels = 1,
        .num_out_channels = 1,
        .input = {
            .d1 = {
                .input_size = input_size
            }
        },
        .output = {
            .d1 = {
                .output_size = output_size
            }
        },
        .layer = {
            .mlp_layer = {
                .weights = weights,
                .biases = biases,

                .weight_grads = weight_grads,
                .bias_grads = bias_grads
            }
        }
    };
    return 0;
}

__global__ void mlp_forward(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int neuron_idx = idx / layer.input.d1.input_size;
    int input_idx = idx % layer.input.d1.input_size;
    int weight_idx = idx;

    if(neuron_idx >= layer.output.d1.output_size) {
        return;
    }

    atomicAdd(&(layer.output.d1.output[neuron_idx]), layer.input.d1.input[input_idx] * layer.layer.mlp_layer.weights[weight_idx]);
}


__global__ void zero_grads_mlp_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int neuron_idx = idx / layer.input.d1.input_size;
    int input_idx = idx % layer.input.d1.input_size;
    int weight_idx = idx;

    if(idx >= layer.input.d1.input_size * layer.output.d1.output_size) {
        return;
    }

    if(input_idx == 0) {
        layer.layer.mlp_layer.bias_grads[neuron_idx] = (DATA_TYPE)0.0;
    }
    layer.layer.mlp_layer.weight_grads[weight_idx] = (DATA_TYPE)0.0;
}

__global__ void zero_input_grads_mlp_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= layer.input.d1.input_size) {
        return;
    }

    layer.input.d1.grads[idx] = (DATA_TYPE)0.0;
}

__global__ void grad_mlp_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int neuron_idx = idx / layer.input.d1.input_size;
    int input_idx = idx % layer.input.d1.input_size;
    int weight_idx = idx;

    if(idx >= layer.input.d1.input_size * layer.output.d1.output_size) {
        return;
    }

    if(input_idx == 0) {
        layer.layer.mlp_layer.bias_grads[neuron_idx] += layer.output.d1.grads[neuron_idx];
    }
    layer.layer.mlp_layer.weight_grads[weight_idx] += layer.output.d1.grads[neuron_idx] * layer.input.d1.input[input_idx];

    if(layer.input.d1.grads != NULL) {
        atomicAdd(&(layer.input.d1.grads[input_idx]), layer.output.d1.grads[neuron_idx] * layer.layer.mlp_layer.weights[weight_idx]);
    }
}

__global__ void update_mlp_layer(Layer layer, DATA_TYPE learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int neuron_idx = idx / layer.input.d1.input_size;
    int input_idx = idx % layer.input.d1.input_size;
    int weight_idx = idx;

    if(idx >= layer.input.d1.input_size * layer.output.d1.output_size) {
        return;
    }

    if(input_idx == 0) {
        layer.layer.mlp_layer.biases[neuron_idx] -= learning_rate * layer.layer.mlp_layer.bias_grads[neuron_idx];
    }
    layer.layer.mlp_layer.weights[weight_idx] -= learning_rate * layer.layer.mlp_layer.weight_grads[weight_idx];
}

int save_mlp_layer(Layer layer, FILE* file) {
    DATA_TYPE* host_weights = (DATA_TYPE*)malloc(layer.input.d1.input_size * layer.output.d1.output_size * sizeof(DATA_TYPE));
    DATA_TYPE* host_biases = (DATA_TYPE*)malloc(layer.output.d1.output_size * sizeof(DATA_TYPE));

    cudaMemcpy(host_weights, layer.layer.mlp_layer.weights, layer.input.d1.input_size * layer.output.d1.output_size * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_biases, layer.layer.mlp_layer.biases, layer.output.d1.output_size * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

    fwrite(host_weights, sizeof(DATA_TYPE), layer.input.d1.input_size * layer.output.d1.output_size, file);
    fwrite(host_biases, sizeof(DATA_TYPE), layer.output.d1.output_size, file);

    return 0;
}

int load_mlp_layer(Layer* layer, FILE* file) {
    DATA_TYPE* host_weights = (DATA_TYPE*)malloc(layer->input.d1.input_size * layer->output.d1.output_size * sizeof(DATA_TYPE));
    DATA_TYPE* host_biases = (DATA_TYPE*)malloc(layer->output.d1.output_size * sizeof(DATA_TYPE));

    fread(host_weights, sizeof(DATA_TYPE), layer->input.d1.input_size * layer->output.d1.output_size, file);
    fread(host_biases, sizeof(DATA_TYPE), layer->output.d1.output_size, file);

    cudaMemcpy(layer->layer.mlp_layer.weights, host_weights, layer->input.d1.input_size * layer->output.d1.output_size * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(layer->layer.mlp_layer.biases, host_biases, layer->output.d1.output_size * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    return 0;
}
