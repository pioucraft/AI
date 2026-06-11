#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <stdio.h>

#include "convolution.h"
#include "nn.h"
#include "utils.h"

int create_convolution_layer(Layer* layer, int input_dimensions, int output_dimensions, int filter_dimensions, int num_filters, int in_channels, int out_channels) {
    DATA_TYPE* filters;
    DATA_TYPE* biases;

    cudaMalloc(&filters, num_filters * filter_dimensions * filter_dimensions * in_channels * sizeof(DATA_TYPE));
    cudaMalloc(&biases, num_filters * sizeof(DATA_TYPE));

    DATA_TYPE deviation = sqrt(2.0 / (filter_dimensions * filter_dimensions * in_channels));
    for(int i = 0; i < num_filters * filter_dimensions * filter_dimensions * in_channels; i++) {
        DATA_TYPE filter = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * deviation * 2 - deviation);
        cudaMemcpy(filters + i, &filter, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    }

    for(int i = 0; i < num_filters; i++) {
        DATA_TYPE bias = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * deviation * 2 - deviation);
        cudaMemcpy(biases + i, &bias, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    }

    DATA_TYPE* filter_grads;
    DATA_TYPE* bias_grads;

    cudaMalloc(&filter_grads, num_filters * filter_dimensions * filter_dimensions * in_channels * sizeof(DATA_TYPE));
    cudaMalloc(&bias_grads, num_filters * sizeof(DATA_TYPE));

    *layer = {
        .layer_type = LAYER_TYPE_CONVOLUTION,
        .num_in_channels = in_channels,
        .num_out_channels = out_channels,
        .input = {
            .d2 = {
                .input_dimensions = input_dimensions 
            }
        },
        .output = {
            .d2 = {
                .output_dimensions = output_dimensions
            }
        },
        .layer = {
            .convolution_layer = {
                .filter_dimensions = filter_dimensions,
                .filters_num = num_filters,
                .filters = filters,
                .biases = biases,
                .filter_grads = filter_grads,
                .bias_grads = bias_grads
            }
        }
    };
    return 0;
}


__global__ void convolution_forward(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int output_channel = idx / (layer.output.d2.output_dimensions * layer.output.d2.output_dimensions);
    int filter = output_channel;

    int output_x = idx % layer.output.d2.output_dimensions;
    int output_y = (idx / layer.output.d2.output_dimensions) % layer.output.d2.output_dimensions;
    
    int output_channel_offset = output_channel * layer.output.d2.output_dimensions * layer.output.d2.output_dimensions;
    int filter_offset = filter * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions * layer.num_in_channels;

    if(idx >= layer.num_out_channels * layer.output.d2.output_dimensions * layer.output.d2.output_dimensions) {
        return;
    }

    int output_location = output_channel_offset + output_y * layer.output.d2.output_dimensions + output_x;
    layer.output.d2.output[output_location] = layer.layer.convolution_layer.biases[filter];
    for(int input_channel = 0; input_channel < layer.num_in_channels; input_channel++) {

        int input_channel_offset = input_channel * layer.input.d2.input_dimensions * layer.input.d2.input_dimensions;
        
        for(int x = 0; x < layer.layer.convolution_layer.filter_dimensions; x++) {
            for(int y = 0; y < layer.layer.convolution_layer.filter_dimensions; y++) {
                int input_x = output_x + x;
                int input_y = output_y + y;

                int filter_channel_offset = filter_offset + input_channel * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions;

                DATA_TYPE input_value = layer.input.d2.input[input_channel_offset + input_y * layer.input.d2.input_dimensions + input_x];
                DATA_TYPE filter_value = layer.layer.convolution_layer.filters[filter_channel_offset + y * layer.layer.convolution_layer.filter_dimensions + x];

                layer.output.d2.output[output_location] += input_value * filter_value;
            }
        }
    }
}


__global__ void zero_grads_convolution_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = idx / (layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions);

    if(idx >= layer.layer.convolution_layer.filters_num * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions) {
        return;
    }

    if(idx % (layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions) == 0) {
        layer.layer.convolution_layer.bias_grads[filter] = (DATA_TYPE)0.0;
    }
    for(int in_channel = 0; in_channel < layer.num_in_channels; in_channel++) {
        int filter_idx = filter * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions * layer.num_in_channels + in_channel * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions + idx % (layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions);
        layer.layer.convolution_layer.filter_grads[filter_idx] = (DATA_TYPE)0.0;
    }
}

__global__ void zero_input_grads_convolution_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= layer.num_in_channels * layer.input.d2.input_dimensions * layer.input.d2.input_dimensions) {
        return;
    }

    layer.input.d2.grads[idx] = (DATA_TYPE)0.0;
}

__global__ void grad_convolution_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_channel = idx / (layer.output.d2.output_dimensions * layer.output.d2.output_dimensions);
    int filter = output_channel;

    int output_x = idx % layer.output.d2.output_dimensions;
    int output_y = (idx / layer.output.d2.output_dimensions) % layer.output.d2.output_dimensions;
    
    int output_channel_offset = output_channel * layer.output.d2.output_dimensions * layer.output.d2.output_dimensions;
    int filter_offset = filter * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions * layer.num_in_channels;

    if(idx >= layer.num_out_channels * layer.output.d2.output_dimensions * layer.output.d2.output_dimensions) {
        return;
    }

    atomicAdd(&(layer.layer.convolution_layer.bias_grads[filter]), layer.output.d2.grads[output_channel_offset + output_y * layer.output.d2.output_dimensions + output_x]);

    for(int input_channel = 0; input_channel < layer.num_in_channels; input_channel++) {

        int input_channel_offset = input_channel * layer.input.d2.input_dimensions * layer.input.d2.input_dimensions;

        for(int x = 0; x < layer.layer.convolution_layer.filter_dimensions; x++) {
            for(int y = 0; y < layer.layer.convolution_layer.filter_dimensions; y++) {
                int input_x = output_x + x;
                int input_y = output_y + y;

                DATA_TYPE input_value = layer.input.d2.input[input_channel_offset + input_y * layer.input.d2.input_dimensions + input_x];
                DATA_TYPE grad_value = layer.output.d2.grads[output_channel_offset + output_y * layer.output.d2.output_dimensions + output_x];

                int filter_channel_offset = filter_offset + input_channel * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions;

                atomicAdd(&(layer.layer.convolution_layer.filter_grads[filter_channel_offset + y * layer.layer.convolution_layer.filter_dimensions + x]), grad_value * input_value);

                if(layer.input.d2.grads != NULL) {
                    atomicAdd(&(layer.input.d2.grads[input_channel_offset + input_y * layer.input.d2.input_dimensions + input_x]), grad_value * layer.layer.convolution_layer.filters[filter_channel_offset + y * layer.layer.convolution_layer.filter_dimensions + x]);
                }

            }
        }
    }


}

__global__ void update_convolution_layer(Layer layer, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = idx / (layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions);

    if(idx >= layer.layer.convolution_layer.filters_num * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions) {
        return;
    }

    if(idx % (layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions) == 0) {
        layer.layer.convolution_layer.biases[filter] -= learning_rate * layer.layer.convolution_layer.bias_grads[filter];
    }

    for(int in_channel = 0; in_channel < layer.num_in_channels; in_channel++) {
        int filter_idx = filter * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions * layer.num_in_channels + in_channel * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions + idx % (layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions);
        layer.layer.convolution_layer.filters[filter_idx] -= learning_rate * layer.layer.convolution_layer.filter_grads[filter_idx];
    }
}

int save_convolution_layer(Layer layer, FILE* file) {
    DATA_TYPE* host_filters = (DATA_TYPE*)malloc(layer.layer.convolution_layer.filters_num * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions * layer.num_in_channels * sizeof(DATA_TYPE));
    DATA_TYPE* host_biases = (DATA_TYPE*)malloc(layer.layer.convolution_layer.filters_num * sizeof(DATA_TYPE));

    cudaMemcpy(host_filters, layer.layer.convolution_layer.filters, layer.layer.convolution_layer.filters_num * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions * layer.num_in_channels * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_biases, layer.layer.convolution_layer.biases, layer.layer.convolution_layer.filters_num * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

    fwrite(host_filters, sizeof(DATA_TYPE), layer.layer.convolution_layer.filters_num * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions * layer.num_in_channels, file);
    fwrite(host_biases, sizeof(DATA_TYPE), layer.layer.convolution_layer.filters_num, file);

    return 0;
}

int load_convolution_layer(Layer* layer, FILE* file) {
    DATA_TYPE* host_filters = (DATA_TYPE*)malloc(layer->layer.convolution_layer.filters_num * layer->layer.convolution_layer.filter_dimensions * layer->layer.convolution_layer.filter_dimensions * layer->num_in_channels * sizeof(DATA_TYPE));
    DATA_TYPE* host_biases = (DATA_TYPE*)malloc(layer->layer.convolution_layer.filters_num * sizeof(DATA_TYPE));

    fread(host_filters, sizeof(DATA_TYPE), layer->layer.convolution_layer.filters_num * layer->layer.convolution_layer.filter_dimensions * layer->layer.convolution_layer.filter_dimensions * layer->num_in_channels, file);
    fread(host_biases, sizeof(DATA_TYPE), layer->layer.convolution_layer.filters_num, file);

    cudaMemcpy(layer->layer.convolution_layer.filters, host_filters, layer->layer.convolution_layer.filters_num * layer->layer.convolution_layer.filter_dimensions * layer->layer.convolution_layer.filter_dimensions * layer->num_in_channels * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(layer->layer.convolution_layer.biases, host_biases, layer->layer.convolution_layer.filters_num * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    return 0;
}
