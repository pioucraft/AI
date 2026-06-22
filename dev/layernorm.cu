#include <cuda_runtime.h>
#include <device_atomic_functions.h>

#include "nn.h"
#include "utils.h"
#include "layernorm.h"

#define EPSILON 1e-5

// TODO : Make this thing work for tensors of layer rank > 2
int create_layernorm_layer(Layer* layer, int tensor_rank, int tensor_dimensions[TENSOR_MAX_RANK]) {
    DATA_TYPE* gains;
    DATA_TYPE* biases;

    int input_size = 1;
    for(int i = 0; i < tensor_rank; i++) {
        input_size *= tensor_dimensions[i];
    }

    int gain_bias_size = input_size / tensor_dimensions[0];

    cudaMalloc(&gains, gain_bias_size * sizeof(DATA_TYPE));
    cudaMalloc(&biases, gain_bias_size * sizeof(DATA_TYPE));

    for(int i = 0; i < gain_bias_size; i++) {
        DATA_TYPE gain = (DATA_TYPE)1.0;
        DATA_TYPE bias = (DATA_TYPE)0.0;
        
        cudaMemcpy(gains + i, &gain, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
        cudaMemcpy(biases + i, &bias, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    }

    DATA_TYPE* gain_grads;
    DATA_TYPE* bias_grads;

    cudaMalloc(&gain_grads, gain_bias_size * sizeof(DATA_TYPE));
    cudaMalloc(&bias_grads, gain_bias_size * sizeof(DATA_TYPE));

    DATA_TYPE* means;
    DATA_TYPE* variances;

    cudaMalloc(&means, sizeof(DATA_TYPE) * tensor_dimensions[0]);
    cudaMalloc(&variances, sizeof(DATA_TYPE) * tensor_dimensions[0]);

    DATA_TYPE* mean_grads;
    DATA_TYPE* variance_grads;

    cudaMalloc(&mean_grads, sizeof(DATA_TYPE) * tensor_dimensions[0]);
    cudaMalloc(&variance_grads, sizeof(DATA_TYPE) * tensor_dimensions[0]);

    DATA_TYPE* normalized_values;

    cudaMalloc(&normalized_values, input_size * sizeof(DATA_TYPE));

    *layer = (Layer){
        .layer_type = LAYER_TYPE_LAYERNORM,
        .num_in_channels = 1,
        .num_out_channels = 1,
        .input = {
            .tensor = {
                .tensor_rank = tensor_rank,
                .input_size = input_size
            }
        },
        .output = {
            .tensor = {
                .tensor_rank = tensor_rank,
                .output_size = input_size
            }
        },
        .layer = {
            .layernorm_layer = {
                .gains = gains,
                .biases = biases,

                .gain_grads = gain_grads,
                .bias_grads = bias_grads,

                .means = means,
                .variances = variances,

                .mean_grads = mean_grads,
                .variance_grads = variance_grads,

                .normalized_values = normalized_values,
            }
        }
    };
    memcpy(layer->input.tensor.tensor_dimensions, tensor_dimensions, TENSOR_MAX_RANK * sizeof(int));
    memcpy(layer->output.tensor.tensor_dimensions, tensor_dimensions, TENSOR_MAX_RANK * sizeof(int));

    return 0;
}

__global__ void layernorm_forward_zero_variance_mean(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= layer.input.tensor.tensor_dimensions[0]) {
        return;
    }

    layer.layer.layernorm_layer.means[idx] = (DATA_TYPE)0.0;
    layer.layer.layernorm_layer.variances[idx] = (DATA_TYPE)0.0;
}

__global__ void layernorm_forward_mean(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int input_size = layer.input.tensor.input_size;
    if(idx >= input_size) {
        return;
    }

    int vector_size = layer.input.tensor.tensor_dimensions[0];
    int vector_idx = idx / vector_size;
    atomicAdd(&(layer.layer.layernorm_layer.means[vector_idx]), layer.input.tensor.input[idx] / vector_size);
}

__global__ void layernorm_forward_variance(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int input_size = layer.input.tensor.input_size;
    if(idx >= input_size) {
        return;
    }

    int vector_size = layer.input.tensor.tensor_dimensions[0];
    int vector_idx = idx / vector_size;
    DATA_TYPE mean = layer.layer.layernorm_layer.means[vector_idx];
    DATA_TYPE value = layer.input.tensor.input[idx] - mean;
    atomicAdd(&(layer.layer.layernorm_layer.variances[vector_idx]), value * value / vector_size);
}

__global__ void layernorm_forward(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int input_size = layer.input.tensor.input_size;

    if(idx >= input_size) {
        return;
    }

    int vector_size = layer.input.tensor.tensor_dimensions[0];
    int vector_idx = idx / vector_size;
    int element_idx = idx % vector_size;
    DATA_TYPE mean = layer.layer.layernorm_layer.means[vector_idx];
    DATA_TYPE variance = layer.layer.layernorm_layer.variances[vector_idx];
    DATA_TYPE gain = layer.layer.layernorm_layer.gains[element_idx];
    DATA_TYPE bias = layer.layer.layernorm_layer.biases[element_idx];
    
    DATA_TYPE normalized_value = (layer.input.tensor.input[idx] - mean) / sqrt(variance + EPSILON);
    layer.layer.layernorm_layer.normalized_values[idx] = normalized_value;

    layer.output.tensor.output[idx] = gain * normalized_value + bias;
}

__global__ void zero_input_grads_layernorm_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int input_size = layer.input.tensor.input_size;

    if(idx >= input_size) {
        return;
    }

    layer.input.tensor.grads[idx] = (DATA_TYPE)0.0;
    if(idx % layer.input.tensor.tensor_dimensions[0] == 0) {
        layer.layer.layernorm_layer.mean_grads[idx / layer.input.tensor.tensor_dimensions[0]] = (DATA_TYPE)0.0;
        layer.layer.layernorm_layer.variance_grads[idx / layer.input.tensor.tensor_dimensions[0]] = (DATA_TYPE)0.0;
    }
}

__global__ void grad_layernorm_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= layer.input.tensor.input_size) {
        return;
    }

    int vector_size = layer.input.tensor.tensor_dimensions[0];
    int vector_idx = idx / vector_size;
    int element_idx = idx % vector_size;

    DATA_TYPE mean = layer.layer.layernorm_layer.means[vector_idx];
    DATA_TYPE variance = layer.layer.layernorm_layer.variances[vector_idx];
    DATA_TYPE gain = layer.layer.layernorm_layer.gains[element_idx];
    DATA_TYPE bias = layer.layer.layernorm_layer.biases[element_idx];
    DATA_TYPE normalized_value = layer.layer.layernorm_layer.normalized_values[idx];

    atomicAdd(&(layer.layer.layernorm_layer.bias_grads[element_idx]), layer.output.tensor.grads[idx]);
    atomicAdd(&(layer.layer.layernorm_layer.gain_grads[element_idx]), layer.output.tensor.grads[idx] * normalized_value);

    if(layer.input.tensor.grads != NULL) {
        DATA_TYPE grad_output = layer.output.tensor.grads[idx];
        DATA_TYPE grad_normalized = grad_output * gain;

        DATA_TYPE grad_input_through_normalized = grad_normalized / sqrt(variance + EPSILON);
        atomicAdd(&(layer.input.tensor.grads[idx]), grad_input_through_normalized);

        DATA_TYPE grad_mean = -grad_normalized / sqrt(variance + EPSILON);
        atomicAdd(&(layer.layer.layernorm_layer.mean_grads[vector_idx]), grad_mean);

        DATA_TYPE grad_variance = grad_normalized * (layer.input.tensor.input[idx] - mean) * -0.5 * pow(variance + EPSILON, -1.5);
        atomicAdd(&(layer.layer.layernorm_layer.variance_grads[vector_idx]), grad_variance);
    }
}

__global__ void grad_layernorm_layer_step_two(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= layer.input.tensor.input_size) {
        return;
    }

    int vector_size = layer.input.tensor.tensor_dimensions[0];
    int vector_idx = idx / vector_size;
    int element_idx = idx % vector_size;

    DATA_TYPE mean = layer.layer.layernorm_layer.means[vector_idx];
    DATA_TYPE variance = layer.layer.layernorm_layer.variances[vector_idx];
    DATA_TYPE gain = layer.layer.layernorm_layer.gains[element_idx];
    DATA_TYPE bias = layer.layer.layernorm_layer.biases[element_idx];
    DATA_TYPE normalized_value = layer.layer.layernorm_layer.normalized_values[idx];

    if(layer.input.tensor.grads != NULL) {
        DATA_TYPE grad_input_through_mean = layer.layer.layernorm_layer.mean_grads[vector_idx] / vector_size;
        atomicAdd(&(layer.input.tensor.grads[idx]), grad_input_through_mean);

        DATA_TYPE grad_input_through_variance = layer.layer.layernorm_layer.variance_grads[vector_idx] * 2.0 * (layer.input.tensor.input[idx] - mean) / vector_size;
        atomicAdd(&(layer.input.tensor.grads[idx]), grad_input_through_variance);
    }

}

__global__ void zero_grads_layernorm_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= layer.input.tensor.input_size) {
        return;
    }

    int vector_size = layer.input.tensor.tensor_dimensions[0];
    int vector_idx = idx / vector_size;
    int element_idx = idx % layer.input.tensor.tensor_dimensions[0];

    if(vector_idx == 0) {
        layer.layer.layernorm_layer.gain_grads[element_idx] = (DATA_TYPE)0.0;
        layer.layer.layernorm_layer.bias_grads[element_idx] = (DATA_TYPE)0.0;
    }

    if(element_idx == 0) {
        layer.layer.layernorm_layer.mean_grads[vector_idx] = (DATA_TYPE)0.0;
        layer.layer.layernorm_layer.variance_grads[vector_idx] = (DATA_TYPE)0.0;
    }
}

__global__ void update_layernorm_layer(Layer layer, DATA_TYPE learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int vector_size = layer.input.tensor.tensor_dimensions[0];
    if(idx >= vector_size) {
        return;
    }

    layer.layer.layernorm_layer.gains[idx] -= learning_rate * layer.layer.layernorm_layer.gain_grads[idx];
    layer.layer.layernorm_layer.biases[idx] -= learning_rate * layer.layer.layernorm_layer.bias_grads[idx];
}

// TODO : Implement save and load
