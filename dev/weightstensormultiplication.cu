#include "utils.h"
#include "weightstensormultiplication.h"

// This works for only rank2 tensors basically...
int create_weightstensormultiplication(Layer* layer, int tensor_rank, int tensor_dimensions[TENSOR_MAX_RANK], int weights_rank, int weights_dimensions[TENSOR_MAX_RANK]) {
    int input_size = 1;
    for (int i = 0; i < tensor_rank; i++) {
        input_size *= tensor_dimensions[i];
    }

    int output_size = tensor_dimensions[0] * weights_dimensions[1];

    int weights_size = 1;
    for (int i = 0; i < weights_rank; i++) {
        weights_size *= weights_dimensions[i];
    }

    DATA_TYPE* weights;
    DATA_TYPE* weight_grads;

    cudaMalloc(&weights, weights_size * sizeof(DATA_TYPE));
    cudaMalloc(&weight_grads, weights_size * sizeof(DATA_TYPE));

    DATA_TYPE deviation = sqrt(2.0 / (weights_dimensions[0] + weights_dimensions[1]));

    for(int i = 0; i < weights_size; i++) {
        DATA_TYPE weight = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * deviation * 2 - deviation);
        cudaMemcpy(weights + i, &weight, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    }

    *layer = (Layer){
        .layer_type = LAYER_TYPE_WEIGHTSTENSORMULTIPLICATION,
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
                .tensor_rank = 2,
                .output_size = output_size
            }
        },
        .layer = {
            .weightstensormultiplication_layer = {
                .weights = weights,
                .weight_grads = weight_grads,
                .weights_rank = weights_rank,
            }
        }
    };

    memcpy(layer->layer.weightstensormultiplication_layer.weights_dimensions, weights_dimensions, sizeof(int) * 2);
    memcpy(layer->input.tensor.tensor_dimensions, tensor_dimensions, sizeof(int) * 2);
    memcpy(layer->output.tensor.tensor_dimensions, (int[]){tensor_dimensions[0], weights_dimensions[1]}, sizeof(int) * 2);

    return 0;
}
