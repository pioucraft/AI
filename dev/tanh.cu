#include <cuda_runtime.h>
#include <device_atomic_functions.h>

#include "nn.h"
#include "utils.h"
#include "tanh.h"

int create_tanh_layer(Layer* layer, int input_size) {
    *layer = {
        .layer_type = LAYER_TYPE_TANH,
        .num_in_channels = 1,
        .num_out_channels = 1,
        .input = {
            .d1 = {
                .input_size = input_size
            }
        },
        .output = {
            .d1 = {
                .output_size = input_size
            }
        },
    };
    return 0;
}

__global__ void tanh_forward(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < layer.output.d1.output_size) {
        layer.output.d1.output[idx] = tanh(layer.input.d1.input[idx]);
    }
}

__global__ void zero_input_grads_tanh_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < layer.input.d1.input_size) {
        layer.input.d1.grads[idx] = (DATA_TYPE)0.0;
    }
}

__global__ void grad_tanh_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= layer.input.d1.input_size) {
        return;
    }

    DATA_TYPE output = layer.output.d1.output[idx];
    layer.input.d1.grads[idx] = (1 - output * output) * layer.output.d1.grads[idx];
}

