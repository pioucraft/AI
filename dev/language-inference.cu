#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mlp.h"
#include "nn.h"
#include "gelu.h"
#include "utils.h"
#include "attention.h"
#include "layernorm.h"
#include "softmax.h"
#include "../language/language.h"

#define CONTEXT_LENGTH 128
#define TOKENS_SIZE 65
#define EMBEDDING_SIZE 128
#define QUERY_KEY_SIZE 16
#define NUM_HEADS 8
#define FFN_HIDDEN_SIZE 512
#define DATASET_SIZE 1000000

__device__ unsigned int lcg_next(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

__global__ void sample_token(unsigned int* rng_state, DATA_TYPE* probs, DATA_TYPE* logits,
                             int context_length, int tokens_size, int* output_token,
                             DATA_TYPE temperature) {
    DATA_TYPE* last_probs = probs + (context_length - 1) * tokens_size;
    DATA_TYPE* last_logits = logits + (context_length - 1) * tokens_size;
    int token;

    if(temperature == 0.0f) {
        DATA_TYPE max_val = last_logits[0];
        token = 0;
        for(int i = 1; i < tokens_size; i++) {
            if(last_logits[i] > max_val) {
                max_val = last_logits[i];
                token = i;
            }
        }
    } else {
        DATA_TYPE r = (DATA_TYPE)lcg_next(rng_state) / (DATA_TYPE)UINT_MAX;
        DATA_TYPE cumulative = 0.0;
        token = tokens_size - 1;
        for(int i = 0; i < tokens_size; i++) {
            cumulative += last_probs[i];
            if(cumulative >= r) {
                token = i;
                break;
            }
        }
    }

    *output_token = token;
}

__global__ void one_hot_encode(DATA_TYPE* context, int context_length, int tokens_size, int token) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= tokens_size) return;
    context[(context_length - 1) * tokens_size + idx] = (idx == token) ? 1.0 : 0.0;
}

int build_nn(NN* nn, Layer** layers_out) {
    int l = 0;
    int num_layers = 10;
    Layer* layers = (Layer*)malloc(sizeof(Layer) * num_layers);

    create_mlp_layer(&layers[l++], 2, (int[]){CONTEXT_LENGTH, TOKENS_SIZE}, EMBEDDING_SIZE);
    create_layernorm_layer(&layers[l++], 2, (int[]){CONTEXT_LENGTH, EMBEDDING_SIZE});
    create_attention_layer(&layers[l++], CONTEXT_LENGTH, EMBEDDING_SIZE, QUERY_KEY_SIZE, NUM_HEADS);
    create_layernorm_layer(&layers[l++], 2, (int[]){CONTEXT_LENGTH, EMBEDDING_SIZE});
    create_mlp_layer(&layers[l++], 2, (int[]){CONTEXT_LENGTH, EMBEDDING_SIZE}, FFN_HIDDEN_SIZE);
    create_gelu_layer(&layers[l++], CONTEXT_LENGTH * FFN_HIDDEN_SIZE);
    create_mlp_layer(&layers[l++], 2, (int[]){CONTEXT_LENGTH, FFN_HIDDEN_SIZE}, EMBEDDING_SIZE);
    create_layernorm_layer(&layers[l++], 2, (int[]){CONTEXT_LENGTH, EMBEDDING_SIZE});
    create_mlp_layer(&layers[l++], 2, (int[]){CONTEXT_LENGTH, EMBEDDING_SIZE}, TOKENS_SIZE);
    create_softmax_layer(&layers[l++], 2, (int[]){CONTEXT_LENGTH, TOKENS_SIZE}, 1.0);

    int pos_count = CONTEXT_LENGTH * EMBEDDING_SIZE;
    DATA_TYPE* pos_encoding;
    DATA_TYPE* pos_encoding_grads;
    cudaMalloc(&pos_encoding, pos_count * sizeof(DATA_TYPE));
    cudaMalloc(&pos_encoding_grads, pos_count * sizeof(DATA_TYPE));

    *nn = (NN){
        .num_layers = l,
        .layers = layers,
        .adamw_timestep = 0,
        .pos_encoding = pos_encoding,
        .pos_encoding_grads = pos_encoding_grads,
        .pos_encoding_adam = {.m = NULL, .v = NULL},
        .attn_residual_layer_idx = 2,
        .ffn_residual_layer_idx = 6,
        .grad_scale = 0.0f,
    };

    create_nn(nn);
    *layers_out = layers;
    return l;
}

int main(int argc, char* argv[]) {
    DATA_TYPE temperature = 1.0f;
    int num_chars = 256;

    if(argc > 1) temperature = atof(argv[1]);
    if(argc > 2) num_chars = atoi(argv[2]);

    srand(time(NULL));

    NN nn;
    Layer* layers;
    build_nn(&nn, &layers);

    if(load_nn(&nn, "model.data") != 0) {
        fprintf(stderr, "Failed to load model.data\n");
        return 1;
    }

    DATA_TYPE* dataset;
    char* tokens;
    load_language_dataset("language/tinyshakespeare.txt", DATASET_SIZE, &dataset, &tokens);

    int max_pos = DATASET_SIZE - CONTEXT_LENGTH - 1;
    int start_pos = rand() % max_pos;

    DATA_TYPE* host_context = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * CONTEXT_LENGTH * TOKENS_SIZE);
    DATA_TYPE* device_context;
    cudaMalloc(&device_context, sizeof(DATA_TYPE) * CONTEXT_LENGTH * TOKENS_SIZE);

    cudaMemcpy(device_context, dataset + start_pos * TOKENS_SIZE,
               sizeof(DATA_TYPE) * CONTEXT_LENGTH * TOKENS_SIZE, cudaMemcpyDeviceToDevice);
    cudaMemcpy(host_context, device_context,
               sizeof(DATA_TYPE) * CONTEXT_LENGTH * TOKENS_SIZE, cudaMemcpyDeviceToHost);

    char input_text[CONTEXT_LENGTH + 1];
    for(int j = 0; j < CONTEXT_LENGTH; j++) {
        for(int k = 0; k < TOKENS_SIZE; k++) {
            if(host_context[j * TOKENS_SIZE + k] > 0.5f) {
                input_text[j] = untokenizer(k, tokens);
                break;
            }
        }
    }
    input_text[CONTEXT_LENGTH] = '\0';
    free(host_context);

    printf("\x1b[31m=== Input (random %d-char snippet, temperature=%.2f) ===\x1b[0m\n", CONTEXT_LENGTH, temperature);
    printf("%s\n", input_text);
    printf("\x1b[31m=== Generated %d chars ===\x1b[0m\n", num_chars);

    int softmax_idx = nn.num_layers - 1;
    int lm_head_idx = nn.num_layers - 2;
    nn.layers[softmax_idx].layer.softmax_layer.temperature = temperature;

    unsigned int* d_rng_state;
    int* d_predicted_token;
    cudaMalloc(&d_rng_state, sizeof(unsigned int));
    cudaMalloc(&d_predicted_token, sizeof(int));
    unsigned int init_seed = time(NULL);
    cudaMemcpy(d_rng_state, &init_seed, sizeof(unsigned int), cudaMemcpyHostToDevice);

    int encode_blocks = (TOKENS_SIZE + 255) / 256;

    char output_text[num_chars + 1];
    for(int step = 0; step < num_chars; step++) {
        call_nn(&nn, device_context, 0);

        sample_token<<<1, 1>>>(d_rng_state,
            nn.layers[softmax_idx].output.tensor.output,
            nn.layers[lm_head_idx].output.tensor.output,
            CONTEXT_LENGTH, TOKENS_SIZE, d_predicted_token, temperature);
        cudaDeviceSynchronize();

        int predicted_token;
        cudaMemcpy(&predicted_token, d_predicted_token, sizeof(int), cudaMemcpyDeviceToHost);

        output_text[step] = untokenizer(predicted_token, tokens);

        cudaMemcpy(device_context, device_context + TOKENS_SIZE,
                   sizeof(DATA_TYPE) * (CONTEXT_LENGTH - 1) * TOKENS_SIZE,
                   cudaMemcpyDeviceToDevice);

        one_hot_encode<<<encode_blocks, 256>>>(device_context, CONTEXT_LENGTH, TOKENS_SIZE, predicted_token);
        cudaDeviceSynchronize();
    }
    output_text[num_chars] = '\0';

    printf("%s\n", output_text);
    printf("\x1b[31m========================================\x1b[0m\n");

    cudaFree(device_context);
    cudaFree(d_rng_state);
    cudaFree(d_predicted_token);
    cudaFree(nn.pos_encoding);
    cudaFree(nn.pos_encoding_grads);
    free(layers);

    return 0;
}
