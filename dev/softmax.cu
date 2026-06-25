#include "nn.h"
#include "softmax.h"

int create_softmax_layer(Layer* layer, int tensor_rank, int tensor_dimensions[TENSOR_MAX_RANK], DATA_TYPE temperature) {
    int input_size = 1;
    for (int i = 0; i < tensor_rank; i++) {
        input_size *= tensor_dimensions[i];
    }

    DATA_TYPE* exp_values;
    DATA_TYPE* sums_exp_values;
    DATA_TYPE* grad_sums;

    cudaMalloc(&exp_values, input_size * sizeof(DATA_TYPE));
    cudaMalloc(&sums_exp_values, tensor_dimensions[0] * sizeof(DATA_TYPE));
    cudaMalloc(&grad_sums, tensor_dimensions[0] * sizeof(DATA_TYPE));

    *layer = (Layer){
        .layer_type = LAYER_TYPE_SOFTMAX,
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
            .softmax_layer = {
                .temperature = temperature,

                .exp_values = exp_values,
                .sums_exp_values = sums_exp_values,
                .grad_sums = grad_sums
            }
        }
    };
    memcpy(layer->input.tensor.tensor_dimensions, tensor_dimensions, 2 * sizeof(int));
    memcpy(layer->output.tensor.tensor_dimensions, tensor_dimensions, 2 * sizeof(int));

    return 0;
}

__global__ void softmax_zero_exp_sums(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < layer.output.tensor.tensor_dimensions[0]) {
        layer.layer.softmax_layer.sums_exp_values[idx] = 0.0f;
    }
}

__global__ void softmax_compute_exps(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < layer.input.tensor.input_size) {
        int feature_size = layer.output.tensor.tensor_dimensions[1];
        int vector_idx = idx / feature_size;

        DATA_TYPE input_value = layer.input.tensor.input[idx];
        DATA_TYPE exp_value = expf(input_value / layer.layer.softmax_layer.temperature);

        layer.layer.softmax_layer.exp_values[idx] = exp_value;
        atomicAdd(&layer.layer.softmax_layer.sums_exp_values[vector_idx], exp_value);
    }
}

__global__ void softmax_compute_outputs(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < layer.output.tensor.output_size) {
        int vector_idx = idx / layer.output.tensor.tensor_dimensions[1];
        int element_idx = idx % layer.output.tensor.tensor_dimensions[1];

        DATA_TYPE exp_value = layer.layer.softmax_layer.exp_values[idx];
        DATA_TYPE sum_exp_value = layer.layer.softmax_layer.sums_exp_values[vector_idx];

        layer.output.tensor.output[idx] = exp_value / sum_exp_value;
    }
}

__global__ void zero_input_grads_softmax_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < layer.input.tensor.input_size) {
        layer.input.tensor.grads[idx] = 0.0f;
    }
    if (idx < layer.output.tensor.tensor_dimensions[0]) {
        layer.layer.softmax_layer.grad_sums[idx] = 0.0f;
    }
}

__global__ void grad_softmax_layer_step_1(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= layer.output.tensor.output_size) {
        return;
    }

    int feature_size = layer.output.tensor.tensor_dimensions[1];
    int vector_idx = idx / feature_size;

    DATA_TYPE output_value = layer.output.tensor.output[idx];
    DATA_TYPE grad_output_value = layer.output.tensor.grads[idx];
    DATA_TYPE exp_value = layer.layer.softmax_layer.exp_values[idx];
    DATA_TYPE sum_exp_value = layer.layer.softmax_layer.sums_exp_values[vector_idx];

    DATA_TYPE grad_sum = grad_output_value * output_value;
    atomicAdd(&layer.layer.softmax_layer.grad_sums[vector_idx], grad_sum);

    DATA_TYPE grad_input_value_through_output = grad_output_value * exp_value / sum_exp_value * 1.0f / layer.layer.softmax_layer.temperature;
    atomicAdd(&layer.input.tensor.grads[idx], grad_input_value_through_output);
}

__global__ void grad_softmax_layer_step_2(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= layer.input.tensor.input_size) {
        return;
    }

    int feature_size = layer.output.tensor.tensor_dimensions[1];
    int vector_idx = idx / feature_size;

    DATA_TYPE output_value = layer.output.tensor.output[idx];
    DATA_TYPE grad_output_value = layer.output.tensor.grads[idx];
    DATA_TYPE grad_sum = layer.layer.softmax_layer.grad_sums[vector_idx];

    DATA_TYPE grad_input_through_sum = -output_value * grad_sum / layer.layer.softmax_layer.temperature;
    atomicAdd(&layer.input.tensor.grads[idx], grad_input_through_sum);
}
