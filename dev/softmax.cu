#include "nn.h"
#include "softmax.h"

int create_softmax_layer(Layer* layer, int tensor_rank, int tensor_dimensions[TENSOR_MAX_RANK], DATA_TYPE temperature) {
    int input_size = 1;
    for (int i = 0; i < tensor_rank; i++) {
        input_size *= tensor_dimensions[i];
    }

    DATA_TYPE* exp_values;
    DATA_TYPE* sums_exp_values;

    cudaMalloc(&exp_values, input_size * sizeof(DATA_TYPE));
    cudaMalloc(&sums_exp_values, tensor_dimensions[0] * sizeof(DATA_TYPE));

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
                .sums_exp_values = sums_exp_values
            }
        }
    };
    memcpy(layer->input.tensor.tensor_dimensions, tensor_dimensions, TENSOR_MAX_RANK * sizeof(int));
    memcpy(layer->output.tensor.tensor_dimensions, tensor_dimensions, TENSOR_MAX_RANK * sizeof(int));

    return 0;
}
