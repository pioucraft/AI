#include <cuda_runtime.h>
#include <curand_mtgp32.h>
#include <device_atomic_functions.h>
#include <curand_kernel.h>

#include "nn.h"
#include "utils.h"
#include "dropout.h"

__global__ void setup_random_states(curandState_t *state, int seed, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= size) {
        return;
    }

    curand_init(seed, idx, 0, &state[idx]);
}

int create_dropout_layer(Layer* layer, int input_size, DATA_TYPE dropout_rate) {
    curandState_t *random_states;
    cudaMalloc(&random_states, input_size * sizeof(curandState_t));
    setup_random_states<<<input_size / NUM_THREADS + 1, NUM_THREADS>>>(random_states, 42, input_size);
    cudaDeviceSynchronize();
    checkCudaError();

    unsigned char* mask;
    cudaMalloc(&mask, input_size * sizeof(unsigned char));

    *layer = {
        .layer_type = LAYER_TYPE_DROPOUT,
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
        .layer = {
            .dropout_layer = {
                .dropout_rate = dropout_rate,
                .random_states = random_states,
                .mask = mask
            }
        }
    };
    return 0;
}

__global__ void dropout_forward(Layer layer, int run_dropout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= layer.output.d1.output_size) {
        return;
    }

    if(!run_dropout) {
        layer.output.d1.output[idx] = layer.input.d1.input[idx];
        return;
    }

    curandState_t local_state = layer.layer.dropout_layer.random_states[idx];

    DATA_TYPE random_value = curand_uniform(&local_state);
    layer.layer.dropout_layer.random_states[idx] = local_state;

    if(random_value > layer.layer.dropout_layer.dropout_rate) {
        layer.output.d1.output[idx] = 1.0 / (1.0 - layer.layer.dropout_layer.dropout_rate) * layer.input.d1.input[idx];
        layer.layer.dropout_layer.mask[idx] = 1;
    } else {
        layer.output.d1.output[idx] = (DATA_TYPE)0.0;
        layer.layer.dropout_layer.mask[idx] = 0;
    }
}

__global__ void zero_input_grads_dropout_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= layer.input.d1.input_size) {
        return;
    }

    layer.input.d1.grads[idx] = (DATA_TYPE)0.0;
}

__global__ void grad_dropout_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= layer.input.d1.input_size) {
        return;
    }

    if(layer.layer.dropout_layer.mask[idx]) {
        layer.input.d1.grads[idx] = 1.0 / (1.0 - layer.layer.dropout_layer.dropout_rate) * layer.output.d1.grads[idx];
    }
}

