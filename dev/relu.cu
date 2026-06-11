#include <cuda_runtime.h>
#include <device_atomic_functions.h>

#include "nn.h"
#include "utils.h"
#include "relu.h"

int create_relu_layer(Layer* layer, int input_size) {
    *layer = {
        .layer_type = LAYER_TYPE_RELU,
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

__global__ void relu_forward(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= layer.output.d1.output_size) {
        return;
    }

    if(layer.input.d1.input[idx] > 0) {
        layer.output.d1.output[idx] = layer.input.d1.input[idx];
    } else {
        layer.output.d1.output[idx] = (DATA_TYPE)0.0;
    }
}

__global__ void zero_input_grads_relu_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= layer.input.d1.input_size) {
        return;
    }

    layer.input.d1.grads[idx] = (DATA_TYPE)0.0;
}

__global__ void grad_relu_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= layer.input.d1.input_size) {
        return;
    }

    if(layer.input.d1.input[idx] > 0) {
        layer.input.d1.grads[idx] = layer.output.d1.grads[idx];
    }
}

