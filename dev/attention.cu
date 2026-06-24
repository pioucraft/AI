#include "attention.h"

int create_attention_layer(Layer* layer, int context_length, int embedding_size, int query_key_value_size, int num_heads) {
    DATA_TYPE* query_weights;
    DATA_TYPE* key_weights;
    DATA_TYPE* value_weights;

    cudaMalloc(&query_weights, sizeof(DATA_TYPE) * embedding_size * query_key_value_size * num_heads);
    cudaMalloc(&key_weights, sizeof(DATA_TYPE) * embedding_size * query_key_value_size * num_heads);
    cudaMalloc(&value_weights, sizeof(DATA_TYPE) * embedding_size * embedding_size * num_heads);

    DATA_TYPE deviation_query_key = sqrt(2.0 / (DATA_TYPE(embedding_size + query_key_value_size)));
    DATA_TYPE deviation_value = sqrt(2.0 / (DATA_TYPE(embedding_size + embedding_size)));

    // TODO : finish implementing multi headed attention
    for(int i = 0; i < embedding_size * query_key_value_size * num_heads; i++) {
        DATA_TYPE query_weight = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * deviation_query_key * 2 - deviation_query_key);
        cudaMemcpy(query_weights + i, &query_weight, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

        DATA_TYPE key_weight = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * deviation_query_key * 2 - deviation_query_key);
        cudaMemcpy(key_weights + i, &key_weight, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    }

    for(int i = 0; i < embedding_size * embedding_size * num_heads; i++) {
        DATA_TYPE value_weight = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * deviation_value * 2 - deviation_value);
        cudaMemcpy(value_weights + i, &value_weight, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    }

    *layer = (Layer){
        .layer_type = LAYER_TYPE_ATTENTION,
        .num_in_channels = 1,
        .num_out_channels = 1,
        .input = {
            .tensor = {
                .tensor_rank = 2,
                .input_size = context_length * embedding_size,
            }
        },
        .output = {
            .tensor = {
                .tensor_rank = 2,
                .output_size = context_length * embedding_size,
            }
        },
        .layer = {
            .attention_layer = {
                .context_length = context_length,
                .embedding_size = embedding_size,
                .query_key_value_size = query_key_value_size,

                .query_weights = query_weights,
                .key_weights = key_weights,
                .value_weights = value_weights
            }
        }
    };

    memcpy(layer->input.tensor.tensor_dimensions, (int[]){context_length, embedding_size}, 2 * sizeof(int));
    memcpy(layer->output.tensor.tensor_dimensions, (int[]){context_length, embedding_size}, 2 * sizeof(int));

    return 0;
}
