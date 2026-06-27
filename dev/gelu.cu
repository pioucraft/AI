#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <math.h>

#include "nn.h"
#include "utils.h"
#include "gelu.h"

int create_gelu_layer(Layer* layer, int input_size) {
    *layer = {
        .layer_type = LAYER_TYPE_GELU,
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

__global__ void gelu_forward(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= layer.output.d1.output_size) {
        return;
    }

    DATA_TYPE x = layer.input.d1.input[idx];
    layer.output.d1.output[idx] = x * 0.5f * (1.0f + erff(x / sqrtf(2.0f)));
}

__global__ void zero_input_grads_gelu_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= layer.input.d1.input_size) {
        return;
    }

    layer.input.d1.grads[idx] = (DATA_TYPE)0.0;
}

__global__ void grad_gelu_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= layer.input.d1.input_size) {
        return;
    }

    DATA_TYPE x = layer.input.d1.input[idx];
    DATA_TYPE cdf = 0.5f * (1.0f + erff(x / sqrtf(2.0f)));
    DATA_TYPE pdf = expf(-x * x / 2.0f) / sqrtf(2.0f * 3.14159265358979323846f);
    layer.input.d1.grads[idx] = layer.output.d1.grads[idx] * (cdf + x * pdf);
}
