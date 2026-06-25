#include "attention.h"

int create_attention_layer(Layer* layer, int context_length, int embedding_size, int query_key_size, int num_heads) {
    DATA_TYPE* query_weights;
    DATA_TYPE* key_weights;
    DATA_TYPE* value_weights;

    cudaMalloc(&query_weights, sizeof(DATA_TYPE) * embedding_size * query_key_size * num_heads);
    cudaMalloc(&key_weights, sizeof(DATA_TYPE) * embedding_size * query_key_size * num_heads);
    cudaMalloc(&value_weights, sizeof(DATA_TYPE) * embedding_size * embedding_size * num_heads);

    DATA_TYPE deviation_query_key = sqrt(2.0 / (DATA_TYPE(embedding_size + query_key_size)));
    DATA_TYPE deviation_value = sqrt(2.0 / (DATA_TYPE(embedding_size + embedding_size)));

    for(int i = 0; i < embedding_size * query_key_size * num_heads; i++) {
        DATA_TYPE query_weight = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * deviation_query_key * 2 - deviation_query_key);
        cudaMemcpy(query_weights + i, &query_weight, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

        DATA_TYPE key_weight = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * deviation_query_key * 2 - deviation_query_key);
        cudaMemcpy(key_weights + i, &key_weight, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    }

    for(int i = 0; i < embedding_size * embedding_size * num_heads; i++) {
        DATA_TYPE value_weight = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * deviation_value * 2 - deviation_value);
        cudaMemcpy(value_weights + i, &value_weight, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    }

    DATA_TYPE* query_weight_grads;
    DATA_TYPE* key_weight_grads;
    DATA_TYPE* value_weight_grads;

    cudaMalloc(&query_weight_grads, sizeof(DATA_TYPE) * embedding_size * query_key_size * num_heads);
    cudaMalloc(&key_weight_grads, sizeof(DATA_TYPE) * embedding_size * query_key_size * num_heads);
    cudaMalloc(&value_weight_grads, sizeof(DATA_TYPE) * embedding_size * embedding_size * num_heads);

    DATA_TYPE* queries;
    DATA_TYPE* keys;
    DATA_TYPE* values;

    cudaMalloc(&queries, sizeof(DATA_TYPE) * context_length * query_key_size * num_heads);
    cudaMalloc(&keys, sizeof(DATA_TYPE) * context_length * query_key_size * num_heads);
    cudaMalloc(&values, sizeof(DATA_TYPE) * context_length * embedding_size * num_heads);

    DATA_TYPE* attention_scores;
    DATA_TYPE* attention_score_grads;

    cudaMalloc(&attention_scores, sizeof(DATA_TYPE) * context_length * context_length * num_heads);
    cudaMalloc(&attention_score_grads, sizeof(DATA_TYPE) * context_length * context_length * num_heads);

    DATA_TYPE* softmax_exp_values;
    DATA_TYPE* softmax_sums_exp_values;
    DATA_TYPE* softmax_grad_sums;

    cudaMalloc(&softmax_exp_values, context_length * context_length * num_heads * sizeof(DATA_TYPE));
    cudaMalloc(&softmax_sums_exp_values, context_length * num_heads * sizeof(DATA_TYPE));
    cudaMalloc(&softmax_grad_sums, context_length * num_heads * sizeof(DATA_TYPE));

    DATA_TYPE* attention_scores_masked;
    DATA_TYPE* attention_score_masked_grads;

    cudaMalloc(&attention_scores_masked, context_length * context_length * num_heads * sizeof(DATA_TYPE));
    cudaMalloc(&attention_score_masked_grads, context_length * context_length * num_heads * sizeof(DATA_TYPE));

    DATA_TYPE* attention_percentages;
    DATA_TYPE* attention_percentage_grads;

    cudaMalloc(&attention_percentages, context_length * context_length * num_heads * sizeof(DATA_TYPE));
    cudaMalloc(&attention_percentage_grads, context_length * context_length * num_heads * sizeof(DATA_TYPE));

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
                .query_key_size = query_key_size,
                .num_heads = num_heads,

                .query_weights = query_weights,
                .key_weights = key_weights,
                .value_weights = value_weights,

                .query_weight_grads = query_weight_grads,
                .key_weight_grads = key_weight_grads,
                .value_weight_grads = value_weight_grads,

                .queries = queries,
                .keys = keys,
                .values = values,

                .attention_scores = attention_scores,
                .attention_score_grads = attention_score_grads,

                .attention_scores_masked = attention_scores_masked,
                .attention_score_masked_grads = attention_score_masked_grads,

                .softmax_exp_values = softmax_exp_values,
                .softmax_sums_exp_values = softmax_sums_exp_values,
                .softmax_grad_sums = softmax_grad_sums,

                .attention_percentages = attention_percentages,
            }
        }
    };

    memcpy(layer->input.tensor.tensor_dimensions, (int[]){context_length, embedding_size}, 2 * sizeof(int));
    memcpy(layer->output.tensor.tensor_dimensions, (int[]){context_length, embedding_size}, 2 * sizeof(int));

    return 0;
}

__global__ void attention_forward_key_query(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // each idx is an output scalar

    layer.layer.attention_layer.queries[idx] = 0;
    layer.layer.attention_layer.keys[idx] = 0;

    int input_head_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.context_length;
    int output_head_size = layer.layer.attention_layer.query_key_size * layer.layer.attention_layer.context_length;
    int weight_head_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.query_key_size;
    int head_idx = idx / output_head_size;
    int local_idx = idx % output_head_size;

    for(int i = 0; i < layer.layer.attention_layer.embedding_size; i++) {

        int idx_x = local_idx / layer.layer.attention_layer.query_key_size;
        int idx_y = local_idx % layer.layer.attention_layer.query_key_size;

        int input_idx = (idx_x * layer.layer.attention_layer.embedding_size + i);
        int weight_idx = i * layer.layer.attention_layer.query_key_size + idx_y + weight_head_size * head_idx;

        layer.layer.attention_layer.queries[idx] += layer.input.tensor.input[input_idx] * layer.layer.attention_layer.query_weights[weight_idx];
        layer.layer.attention_layer.keys[idx] += layer.input.tensor.input[input_idx] * layer.layer.attention_layer.key_weights[weight_idx];
    }
}

__global__ void attention_forward_value(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // each idx is an output scalar

    layer.layer.attention_layer.values[idx] = 0;

    int input_head_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.context_length;
    int output_head_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.context_length;
    int weight_head_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.embedding_size;
    int head_idx = idx / output_head_size;
    int local_idx = idx % output_head_size;

    for(int i = 0; i < layer.layer.attention_layer.embedding_size; i++) {
        int idx_x = local_idx / layer.layer.attention_layer.embedding_size;
        int idx_y = local_idx % layer.layer.attention_layer.embedding_size;

        int input_idx = (idx_x * layer.layer.attention_layer.embedding_size + i);
        int weight_idx = i * layer.layer.attention_layer.embedding_size + idx_y + weight_head_size * head_idx;

        layer.layer.attention_layer.values[idx] += layer.input.tensor.input[input_idx] * layer.layer.attention_layer.value_weights[weight_idx];
    }
}

__global__ void attention_forward_scores(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each idx is an output scalar
    // Computes the dot product of each query and key vectors

    layer.layer.attention_layer.attention_scores[idx] = 0;

    int input_head_size = layer.layer.attention_layer.query_key_size * layer.layer.attention_layer.context_length;
    int output_head_size = layer.layer.attention_layer.context_length * layer.layer.attention_layer.context_length;
    int head_idx = idx / output_head_size;
    int local_idx = idx % output_head_size;

    for(int i = 0; i < layer.layer.attention_layer.query_key_size; i++) {
        int idx_x = local_idx / layer.layer.attention_layer.context_length;
        int idx_y = local_idx % layer.layer.attention_layer.context_length;

        int query_idx = (idx_x * layer.layer.attention_layer.query_key_size + i) + input_head_size * head_idx;
        int key_idx = (idx_y * layer.layer.attention_layer.query_key_size + i) + input_head_size * head_idx;

        layer.layer.attention_layer.attention_scores[idx] += layer.layer.attention_layer.queries[query_idx] * layer.layer.attention_layer.keys[key_idx];
    }
}

__global__ void attention_forward_masking(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int head_size = layer.layer.attention_layer.context_length * layer.layer.attention_layer.context_length;
    int local_idx = idx % head_size;

    int idx_x = local_idx / layer.layer.attention_layer.context_length;
    int idx_y = local_idx % layer.layer.attention_layer.context_length;

    if(idx_y > idx_x) {
        layer.layer.attention_layer.attention_scores_masked[idx] = -INFINITY;
    } else {
        layer.layer.attention_layer.attention_scores_masked[idx] = layer.layer.attention_layer.attention_scores[idx];
    }
}

__global__ void attention_softmax_zero_exp_sums(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < layer.layer.attention_layer.context_length * layer.layer.attention_layer.num_heads) {
        layer.layer.attention_layer.softmax_sums_exp_values[idx] = 0.0f;
    }
}

__global__ void attention_softmax_compute_exps(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < layer.layer.attention_layer.context_length * layer.layer.attention_layer.context_length * layer.layer.attention_layer.num_heads) {
        int vector_idx = idx / layer.layer.attention_layer.context_length;
        int element_idx = idx % layer.layer.attention_layer.context_length;

        DATA_TYPE input_value = layer.layer.attention_layer.attention_scores_masked[idx];
        DATA_TYPE exp_value = expf(input_value / sqrtf((DATA_TYPE)layer.layer.attention_layer.query_key_size));

        layer.layer.attention_layer.softmax_exp_values[idx] = exp_value;
        atomicAdd(&layer.layer.attention_layer.softmax_sums_exp_values[vector_idx], exp_value);
    }
}

__global__ void attention_softmax_compute_outputs(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < layer.layer.attention_layer.context_length * layer.layer.attention_layer.context_length * layer.layer.attention_layer.num_heads) {
        int vector_idx = idx / layer.layer.attention_layer.context_length;
        int element_idx = idx % layer.layer.attention_layer.context_length;

        DATA_TYPE exp_value = layer.layer.attention_layer.softmax_exp_values[idx];
        DATA_TYPE sum_exp_value = layer.layer.attention_layer.softmax_sums_exp_values[vector_idx];

        layer.layer.attention_layer.attention_percentages[idx] = exp_value / sum_exp_value;
    }
}

__global__ void attention_forward_value_weighted_sum(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each idx is an output scalar

    int idx_x = idx / layer.layer.attention_layer.embedding_size;
    int idx_y = idx % layer.layer.attention_layer.embedding_size;

    if (idx < layer.layer.attention_layer.context_length * layer.layer.attention_layer.embedding_size) {
        layer.output.tensor.output[idx] = layer.input.tensor.input[idx];

        int head_size = layer.layer.attention_layer.context_length * layer.layer.attention_layer.embedding_size;
        
        for(int c_head = 0; c_head < layer.layer.attention_layer.num_heads; c_head++) {
            int head_offset_percentage = c_head * layer.layer.attention_layer.context_length * layer.layer.attention_layer.context_length;
            int head_offset_value = c_head * layer.layer.attention_layer.context_length * layer.layer.attention_layer.embedding_size;

            for(int c_token = 0; c_token < layer.layer.attention_layer.context_length; c_token++) {
                int percentage_idx = head_offset_percentage + idx_x * layer.layer.attention_layer.context_length + c_token;
                int value_idx = head_offset_value + c_token * layer.layer.attention_layer.embedding_size + idx_y;

                layer.output.tensor.output[idx] += layer.layer.attention_layer.attention_percentages[percentage_idx] * layer.layer.attention_layer.values[value_idx];
            }
        }
    }
}
