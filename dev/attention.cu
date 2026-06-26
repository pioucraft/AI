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

    AdamW_State query_adam, key_adam, value_adam;
    int qk_count = embedding_size * query_key_size * num_heads;
    int v_count = embedding_size * embedding_size * num_heads;
    cudaMalloc(&query_adam.m, sizeof(DATA_TYPE) * qk_count);
    cudaMalloc(&query_adam.v, sizeof(DATA_TYPE) * qk_count);
    cudaMemset(query_adam.m, 0, sizeof(DATA_TYPE) * qk_count);
    cudaMemset(query_adam.v, 0, sizeof(DATA_TYPE) * qk_count);
    cudaMalloc(&key_adam.m, sizeof(DATA_TYPE) * qk_count);
    cudaMalloc(&key_adam.v, sizeof(DATA_TYPE) * qk_count);
    cudaMemset(key_adam.m, 0, sizeof(DATA_TYPE) * qk_count);
    cudaMemset(key_adam.v, 0, sizeof(DATA_TYPE) * qk_count);
    cudaMalloc(&value_adam.m, sizeof(DATA_TYPE) * v_count);
    cudaMalloc(&value_adam.v, sizeof(DATA_TYPE) * v_count);
    cudaMemset(value_adam.m, 0, sizeof(DATA_TYPE) * v_count);
    cudaMemset(value_adam.v, 0, sizeof(DATA_TYPE) * v_count);

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

    DATA_TYPE* value_grads;
    DATA_TYPE* query_grads;
    DATA_TYPE* key_grads;

    cudaMalloc(&value_grads, context_length * embedding_size * num_heads * sizeof(DATA_TYPE));
    cudaMalloc(&query_grads, context_length * query_key_size * num_heads * sizeof(DATA_TYPE));
    cudaMalloc(&key_grads, context_length * query_key_size * num_heads * sizeof(DATA_TYPE));

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

                .query_adam = query_adam,
                .key_adam = key_adam,
                .value_adam = value_adam,

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
                .attention_percentage_grads = attention_percentage_grads,

                .value_grads = value_grads,
                .query_grads = query_grads,
                .key_grads = key_grads,
            }
        }
    };

    memcpy(layer->input.tensor.tensor_dimensions, (int[]){context_length, embedding_size}, 2 * sizeof(int));
    memcpy(layer->output.tensor.tensor_dimensions, (int[]){context_length, embedding_size}, 2 * sizeof(int));

    return 0;
}

__global__ void attention_forward_key_query(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = layer.layer.attention_layer.context_length * layer.layer.attention_layer.query_key_size * layer.layer.attention_layer.num_heads;
    if(idx >= total) return;

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

    int total = layer.layer.attention_layer.context_length * layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.num_heads;
    if(idx >= total) return;

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

    int total = layer.layer.attention_layer.context_length * layer.layer.attention_layer.context_length * layer.layer.attention_layer.num_heads;
    if(idx >= total) return;

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

    int total = layer.layer.attention_layer.context_length * layer.layer.attention_layer.context_length * layer.layer.attention_layer.num_heads;
    if(idx >= total) return;

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

__global__ void grad_attention_layer_value_weighted_sum(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each idx is an output scalar

    int idx_x = idx / layer.layer.attention_layer.embedding_size;
    int idx_y = idx % layer.layer.attention_layer.embedding_size;

    if (idx < layer.layer.attention_layer.context_length * layer.layer.attention_layer.embedding_size) {
        atomicAdd(&layer.input.tensor.grads[idx], layer.output.tensor.grads[idx]);

        int head_size = layer.layer.attention_layer.context_length * layer.layer.attention_layer.embedding_size;
        
        for(int c_head = 0; c_head < layer.layer.attention_layer.num_heads; c_head++) {
            int head_offset_percentage = c_head * layer.layer.attention_layer.context_length * layer.layer.attention_layer.context_length;
            int head_offset_value = c_head * layer.layer.attention_layer.context_length * layer.layer.attention_layer.embedding_size;

            for(int c_token = 0; c_token < layer.layer.attention_layer.context_length; c_token++) {
                int percentage_idx = head_offset_percentage + idx_x * layer.layer.attention_layer.context_length + c_token;
                int value_idx = head_offset_value + c_token * layer.layer.attention_layer.embedding_size + idx_y;

                atomicAdd(&layer.layer.attention_layer.value_grads[value_idx], layer.layer.attention_layer.attention_percentages[percentage_idx] * layer.output.tensor.grads[idx]);
                atomicAdd(&layer.layer.attention_layer.attention_percentage_grads[percentage_idx], layer.layer.attention_layer.values[value_idx] * layer.output.tensor.grads[idx]);
            }
        }
    }
}

__global__ void grad_attention_layer_softmax_step_1(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = layer.layer.attention_layer.context_length * layer.layer.attention_layer.context_length * layer.layer.attention_layer.num_heads;
    if (idx >= total) return;

    int vector_idx = idx / layer.layer.attention_layer.context_length;

    DATA_TYPE grad_output = layer.layer.attention_layer.attention_percentage_grads[idx];
    DATA_TYPE output_value = layer.layer.attention_layer.attention_percentages[idx];
    DATA_TYPE exp_value = layer.layer.attention_layer.softmax_exp_values[idx];
    DATA_TYPE sum_exp_value = layer.layer.attention_layer.softmax_sums_exp_values[vector_idx];
    DATA_TYPE temperature = sqrtf((DATA_TYPE)layer.layer.attention_layer.query_key_size);

    DATA_TYPE grad_sum = grad_output * output_value;
    atomicAdd(&layer.layer.attention_layer.softmax_grad_sums[vector_idx], grad_sum);

    DATA_TYPE grad_direct = grad_output * exp_value / sum_exp_value * 1.0f / temperature;
    atomicAdd(&layer.layer.attention_layer.attention_score_masked_grads[idx], grad_direct);
}

__global__ void grad_attention_layer_softmax_step_2(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = layer.layer.attention_layer.context_length * layer.layer.attention_layer.context_length * layer.layer.attention_layer.num_heads;
    if (idx >= total) return;

    int vector_idx = idx / layer.layer.attention_layer.context_length;

    DATA_TYPE grad_output = layer.layer.attention_layer.attention_percentage_grads[idx];
    DATA_TYPE output_value = layer.layer.attention_layer.attention_percentages[idx];
    DATA_TYPE grad_sum = layer.layer.attention_layer.softmax_grad_sums[vector_idx];
    DATA_TYPE temperature = sqrtf((DATA_TYPE)layer.layer.attention_layer.query_key_size);

    DATA_TYPE grad_through_sum = -output_value * grad_sum / temperature;
    atomicAdd(&layer.layer.attention_layer.attention_score_masked_grads[idx], grad_through_sum);
}

__global__ void grad_attention_layer_masking(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = layer.layer.attention_layer.context_length * layer.layer.attention_layer.context_length * layer.layer.attention_layer.num_heads;
    if (idx >= total) return;

    int head_size = layer.layer.attention_layer.context_length * layer.layer.attention_layer.context_length;
    int local_idx = idx % head_size;

    int idx_x = local_idx / layer.layer.attention_layer.context_length;
    int idx_y = local_idx % layer.layer.attention_layer.context_length;

    if (idx_y <= idx_x) {
        layer.layer.attention_layer.attention_score_grads[idx] = layer.layer.attention_layer.attention_score_masked_grads[idx];
    }
}

__global__ void grad_attention_layer_scores(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = layer.layer.attention_layer.context_length * layer.layer.attention_layer.context_length * layer.layer.attention_layer.num_heads;
    if (idx >= total) return;

    int input_head_size = layer.layer.attention_layer.query_key_size * layer.layer.attention_layer.context_length;
    int output_head_size = layer.layer.attention_layer.context_length * layer.layer.attention_layer.context_length;
    int head_idx = idx / output_head_size;
    int local_idx = idx % output_head_size;

    int idx_x = local_idx / layer.layer.attention_layer.context_length;
    int idx_y = local_idx % layer.layer.attention_layer.context_length;

    for (int i = 0; i < layer.layer.attention_layer.query_key_size; i++) {
        int query_idx = (idx_x * layer.layer.attention_layer.query_key_size + i) + input_head_size * head_idx;
        int key_idx = (idx_y * layer.layer.attention_layer.query_key_size + i) + input_head_size * head_idx;

        atomicAdd(&layer.layer.attention_layer.query_grads[query_idx], layer.layer.attention_layer.attention_score_grads[idx] * layer.layer.attention_layer.keys[key_idx]);
        atomicAdd(&layer.layer.attention_layer.key_grads[key_idx], layer.layer.attention_layer.attention_score_grads[idx] * layer.layer.attention_layer.queries[query_idx]);
    }
}

__global__ void grad_attention_layer_key_query(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = layer.layer.attention_layer.context_length * layer.layer.attention_layer.query_key_size * layer.layer.attention_layer.num_heads;
    if (idx >= total) return;

    int input_head_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.context_length;
    int output_head_size = layer.layer.attention_layer.query_key_size * layer.layer.attention_layer.context_length;
    int weight_head_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.query_key_size;
    int head_idx = idx / output_head_size;
    int local_idx = idx % output_head_size;

    int idx_x = local_idx / layer.layer.attention_layer.query_key_size;
    int idx_y = local_idx % layer.layer.attention_layer.query_key_size;

    for (int i = 0; i < layer.layer.attention_layer.embedding_size; i++) {
        int input_idx = (idx_x * layer.layer.attention_layer.embedding_size + i);
        int weight_idx = i * layer.layer.attention_layer.query_key_size + idx_y + weight_head_size * head_idx;

        atomicAdd(&layer.layer.attention_layer.query_weight_grads[weight_idx], layer.layer.attention_layer.query_grads[idx] * layer.input.tensor.input[input_idx]);
        atomicAdd(&layer.layer.attention_layer.key_weight_grads[weight_idx], layer.layer.attention_layer.key_grads[idx] * layer.input.tensor.input[input_idx]);
        atomicAdd(&layer.input.tensor.grads[input_idx], layer.layer.attention_layer.query_grads[idx] * layer.layer.attention_layer.query_weights[weight_idx] + layer.layer.attention_layer.key_grads[idx] * layer.layer.attention_layer.key_weights[weight_idx]);
    }
}

__global__ void grad_attention_layer_value(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = layer.layer.attention_layer.context_length * layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.num_heads;
    if (idx >= total) return;

    int input_head_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.context_length;
    int output_head_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.context_length;
    int weight_head_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.embedding_size;
    int head_idx = idx / output_head_size;
    int local_idx = idx % output_head_size;

    int idx_x = local_idx / layer.layer.attention_layer.embedding_size;
    int idx_y = local_idx % layer.layer.attention_layer.embedding_size;

    for (int i = 0; i < layer.layer.attention_layer.embedding_size; i++) {
        int input_idx = (idx_x * layer.layer.attention_layer.embedding_size + i);
        int weight_idx = i * layer.layer.attention_layer.embedding_size + idx_y + weight_head_size * head_idx;

        atomicAdd(&layer.layer.attention_layer.value_weight_grads[weight_idx], layer.layer.attention_layer.value_grads[idx] * layer.input.tensor.input[input_idx]);
        atomicAdd(&layer.input.tensor.grads[input_idx], layer.layer.attention_layer.value_grads[idx] * layer.layer.attention_layer.value_weights[weight_idx]);
    }
}

__global__ void zero_grads_attention_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int qk_weight_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.query_key_size * layer.layer.attention_layer.num_heads;
    int v_weight_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.num_heads;
    int total = qk_weight_size + qk_weight_size + v_weight_size;

    if (idx >= total) return;

    if (idx < qk_weight_size) {
        layer.layer.attention_layer.query_weight_grads[idx] = 0.0f;
    } else if (idx < 2 * qk_weight_size) {
        layer.layer.attention_layer.key_weight_grads[idx - qk_weight_size] = 0.0f;
    } else {
        layer.layer.attention_layer.value_weight_grads[idx - 2 * qk_weight_size] = 0.0f;
    }
}

__global__ void zero_input_grads_attention_layer(Layer layer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int input_size = layer.input.tensor.input_size;
    int q_size = layer.layer.attention_layer.context_length * layer.layer.attention_layer.query_key_size * layer.layer.attention_layer.num_heads;
    int v_size = layer.layer.attention_layer.context_length * layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.num_heads;
    int scores_size = layer.layer.attention_layer.context_length * layer.layer.attention_layer.context_length * layer.layer.attention_layer.num_heads;
    int sums_size = layer.layer.attention_layer.context_length * layer.layer.attention_layer.num_heads;

    int off = input_size;
    int q_end = off + q_size;
    int v_end = q_end + v_size;
    int score_end = v_end + scores_size;
    int masked_end = score_end + scores_size;
    int pct_end = masked_end + scores_size;
    int total = pct_end + sums_size;

    if (idx >= total) return;

    if (idx < input_size) {
        if (layer.input.tensor.grads != NULL)
            layer.input.tensor.grads[idx] = 0.0f;
    } else if (idx < q_end) {
        layer.layer.attention_layer.query_grads[idx - off] = 0.0f;
        layer.layer.attention_layer.key_grads[idx - off] = 0.0f;
    } else if (idx < v_end) {
        layer.layer.attention_layer.value_grads[idx - q_end] = 0.0f;
    } else if (idx < score_end) {
        layer.layer.attention_layer.attention_score_grads[idx - v_end] = 0.0f;
    } else if (idx < masked_end) {
        layer.layer.attention_layer.attention_score_masked_grads[idx - score_end] = 0.0f;
    } else if (idx < pct_end) {
        layer.layer.attention_layer.attention_percentage_grads[idx - masked_end] = 0.0f;
    } else {
        layer.layer.attention_layer.softmax_grad_sums[idx - pct_end] = 0.0f;
    }
}

__global__ void update_attention_layer(Layer layer, DATA_TYPE learning_rate, int timestep, DATA_TYPE weight_decay) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int qk_weight_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.query_key_size * layer.layer.attention_layer.num_heads;
    int v_weight_size = layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.embedding_size * layer.layer.attention_layer.num_heads;
    int total = qk_weight_size + qk_weight_size + v_weight_size;

    if (idx >= total) return;

    if (idx < qk_weight_size) {
        DATA_TYPE grad = layer.layer.attention_layer.query_weight_grads[idx];
        DATA_TYPE weight = layer.layer.attention_layer.query_weights[idx];
        DATA_TYPE m = layer.layer.attention_layer.query_adam.m[idx];
        DATA_TYPE v = layer.layer.attention_layer.query_adam.v[idx];

        m = ADAMW_BETA1 * m + (1.0f - ADAMW_BETA1) * grad;
        v = ADAMW_BETA2 * v + (1.0f - ADAMW_BETA2) * grad * grad;

        DATA_TYPE m_hat = m / (1.0f - powf(ADAMW_BETA1, timestep));
        DATA_TYPE v_hat = v / (1.0f - powf(ADAMW_BETA2, timestep));

        layer.layer.attention_layer.query_adam.m[idx] = m;
        layer.layer.attention_layer.query_adam.v[idx] = v;

        layer.layer.attention_layer.query_weights[idx] = weight - learning_rate * m_hat / (sqrtf(v_hat) + ADAMW_EPSILON) + learning_rate * weight_decay * weight;
    } else if (idx < 2 * qk_weight_size) {
        int off = idx - qk_weight_size;

        DATA_TYPE grad = layer.layer.attention_layer.key_weight_grads[off];
        DATA_TYPE weight = layer.layer.attention_layer.key_weights[off];
        DATA_TYPE m = layer.layer.attention_layer.key_adam.m[off];
        DATA_TYPE v = layer.layer.attention_layer.key_adam.v[off];

        m = ADAMW_BETA1 * m + (1.0f - ADAMW_BETA1) * grad;
        v = ADAMW_BETA2 * v + (1.0f - ADAMW_BETA2) * grad * grad;

        DATA_TYPE m_hat = m / (1.0f - powf(ADAMW_BETA1, timestep));
        DATA_TYPE v_hat = v / (1.0f - powf(ADAMW_BETA2, timestep));

        layer.layer.attention_layer.key_adam.m[off] = m;
        layer.layer.attention_layer.key_adam.v[off] = v;

        layer.layer.attention_layer.key_weights[off] = weight - learning_rate * m_hat / (sqrtf(v_hat) + ADAMW_EPSILON) + learning_rate * weight_decay * weight;
    } else {
        int off = idx - 2 * qk_weight_size;

        DATA_TYPE grad = layer.layer.attention_layer.value_weight_grads[off];
        DATA_TYPE weight = layer.layer.attention_layer.value_weights[off];
        DATA_TYPE m = layer.layer.attention_layer.value_adam.m[off];
        DATA_TYPE v = layer.layer.attention_layer.value_adam.v[off];

        m = ADAMW_BETA1 * m + (1.0f - ADAMW_BETA1) * grad;
        v = ADAMW_BETA2 * v + (1.0f - ADAMW_BETA2) * grad * grad;

        DATA_TYPE m_hat = m / (1.0f - powf(ADAMW_BETA1, timestep));
        DATA_TYPE v_hat = v / (1.0f - powf(ADAMW_BETA2, timestep));

        layer.layer.attention_layer.value_adam.m[off] = m;
        layer.layer.attention_layer.value_adam.v[off] = v;

        layer.layer.attention_layer.value_weights[off] = weight - learning_rate * m_hat / (sqrtf(v_hat) + ADAMW_EPSILON) + learning_rate * weight_decay * weight;
    }
}

int save_attention_layer(Layer layer, FILE* file) {
    int embedding_size = layer.layer.attention_layer.embedding_size;
    int query_key_size = layer.layer.attention_layer.query_key_size;
    int num_heads = layer.layer.attention_layer.num_heads;

    int qk_weight_size = embedding_size * query_key_size * num_heads;
    int v_weight_size = embedding_size * embedding_size * num_heads;

    DATA_TYPE* host_query_weights = (DATA_TYPE*)malloc(qk_weight_size * sizeof(DATA_TYPE));
    DATA_TYPE* host_key_weights = (DATA_TYPE*)malloc(qk_weight_size * sizeof(DATA_TYPE));
    DATA_TYPE* host_value_weights = (DATA_TYPE*)malloc(v_weight_size * sizeof(DATA_TYPE));

    cudaMemcpy(host_query_weights, layer.layer.attention_layer.query_weights, qk_weight_size * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_key_weights, layer.layer.attention_layer.key_weights, qk_weight_size * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_value_weights, layer.layer.attention_layer.value_weights, v_weight_size * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

    fwrite(host_query_weights, sizeof(DATA_TYPE), qk_weight_size, file);
    fwrite(host_key_weights, sizeof(DATA_TYPE), qk_weight_size, file);
    fwrite(host_value_weights, sizeof(DATA_TYPE), v_weight_size, file);

    free(host_query_weights);
    free(host_key_weights);
    free(host_value_weights);

    return 0;
}

int load_attention_layer(Layer* layer, FILE* file) {
    int embedding_size = layer->layer.attention_layer.embedding_size;
    int query_key_size = layer->layer.attention_layer.query_key_size;
    int num_heads = layer->layer.attention_layer.num_heads;

    int qk_weight_size = embedding_size * query_key_size * num_heads;
    int v_weight_size = embedding_size * embedding_size * num_heads;

    DATA_TYPE* host_query_weights = (DATA_TYPE*)malloc(qk_weight_size * sizeof(DATA_TYPE));
    DATA_TYPE* host_key_weights = (DATA_TYPE*)malloc(qk_weight_size * sizeof(DATA_TYPE));
    DATA_TYPE* host_value_weights = (DATA_TYPE*)malloc(v_weight_size * sizeof(DATA_TYPE));

    fread(host_query_weights, sizeof(DATA_TYPE), qk_weight_size, file);
    fread(host_key_weights, sizeof(DATA_TYPE), qk_weight_size, file);
    fread(host_value_weights, sizeof(DATA_TYPE), v_weight_size, file);

    cudaMemcpy(layer->layer.attention_layer.query_weights, host_query_weights, qk_weight_size * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(layer->layer.attention_layer.key_weights, host_key_weights, qk_weight_size * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(layer->layer.attention_layer.value_weights, host_value_weights, v_weight_size * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    free(host_query_weights);
    free(host_key_weights);
    free(host_value_weights);

    return 0;
}
