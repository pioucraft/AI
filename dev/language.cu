#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>

#include "mlp.h"
#include "nn.h"
#include "gelu.h"
#include "utils.h"
#include "attention.h"
#include "layernorm.h"
#include "softmax.h"
#include "../language/language.h"
#include <math.h>

#define NUM_STEPS 200000
#define DATASET_SIZE 1000000
#define BATCH_SIZE 64
#define LEARNING_RATE 3e-4
#define WEIGHT_DECAY 0.0f

#define CONTEXT_LENGTH 128
#define TOKENS_SIZE 65

__device__ unsigned int lcg_next(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return *state;
}

__global__ void sample_token_gpu(unsigned int* rng_state, DATA_TYPE* probs,
                                  int tokens_size, int* output_token) {
    DATA_TYPE r = (DATA_TYPE)lcg_next(rng_state) / (DATA_TYPE)UINT_MAX;
    DATA_TYPE cumulative = 0.0;
    int token = tokens_size - 1;
    for(int i = 0; i < tokens_size; i++) {
        cumulative += probs[i];
        if(cumulative >= r) {
            token = i;
            break;
        }
    }
    *output_token = token;
}

__global__ void one_hot_encode_gpu(DATA_TYPE* context, int tokens_size, int token) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= tokens_size) return;
    context[(CONTEXT_LENGTH - 1) * tokens_size + idx] = (idx == token) ? 1.0 : 0.0;
}

int test_nn(NN* nn, DATA_TYPE* dataset, char* tokens, int pos) {
    DATA_TYPE* host_context = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * CONTEXT_LENGTH * TOKENS_SIZE);
    DATA_TYPE* device_context;
    cudaMalloc(&device_context, sizeof(DATA_TYPE) * CONTEXT_LENGTH * TOKENS_SIZE);

    cudaMemcpy(device_context, dataset + pos * TOKENS_SIZE,
               sizeof(DATA_TYPE) * CONTEXT_LENGTH * TOKENS_SIZE, cudaMemcpyDeviceToDevice);
    cudaMemcpy(host_context, device_context,
               sizeof(DATA_TYPE) * CONTEXT_LENGTH * TOKENS_SIZE, cudaMemcpyDeviceToHost);

    char input_text[129];
    for(int j = 0; j < CONTEXT_LENGTH; j++) {
        int best = 0;
        for(int k = 1; k < TOKENS_SIZE; k++) {
            if(host_context[j * TOKENS_SIZE + k] > host_context[j * TOKENS_SIZE + best]) best = k;
        }
        input_text[j] = untokenizer(best, tokens);
    }
    input_text[CONTEXT_LENGTH] = '\0';
    free(host_context);

    static unsigned int* d_rng_state = NULL;
    static int* d_predicted_token = NULL;
    if(d_rng_state == NULL) {
        cudaMalloc(&d_rng_state, sizeof(unsigned int));
        cudaMalloc(&d_predicted_token, sizeof(int));
        unsigned int seed = 42;
        cudaMemcpy(d_rng_state, &seed, sizeof(unsigned int), cudaMemcpyHostToDevice);
    }

    int softmax_idx = nn->num_layers - 1;
    int encode_blocks = (TOKENS_SIZE + 255) / 256;

    char output_text[65];
    for(int step = 0; step < 64; step++) {
        call_nn(nn, device_context, 0);

        sample_token_gpu<<<1, 1>>>(d_rng_state,
            nn->layers[softmax_idx].output.tensor.output + (CONTEXT_LENGTH - 1) * TOKENS_SIZE,
            TOKENS_SIZE, d_predicted_token);
        cudaDeviceSynchronize();

        int predicted_token;
        cudaMemcpy(&predicted_token, d_predicted_token, sizeof(int), cudaMemcpyDeviceToHost);
        output_text[step] = untokenizer(predicted_token, tokens);

        cudaMemcpy(device_context, device_context + TOKENS_SIZE,
                   sizeof(DATA_TYPE) * (CONTEXT_LENGTH - 1) * TOKENS_SIZE,
                   cudaMemcpyDeviceToDevice);

        one_hot_encode_gpu<<<encode_blocks, 256>>>(device_context, TOKENS_SIZE, predicted_token);
        cudaDeviceSynchronize();
    }
    output_text[64] = '\0';

    cudaFree(device_context);

    printf("\x1b[31m----------------------------\x1b[0m\n");
    printf("input:\n%s\n\n", input_text);
    printf("output:\n%s\n", output_text);
    printf("\x1b[31m----------------------------\x1b[0m\n");
    return 0;
}

DATA_TYPE compute_cross_entropy_loss(NN* nn, DATA_TYPE* expected) {
    DATA_TYPE* output = nn->layers[nn->num_layers - 1].output.tensor.output;
    int tensor_size = nn->layers[nn->num_layers - 1].output.tensor.output_size;

    DATA_TYPE* host_output = (DATA_TYPE*)malloc(tensor_size * sizeof(DATA_TYPE));
    DATA_TYPE* host_expected = (DATA_TYPE*)malloc(tensor_size * sizeof(DATA_TYPE));

                cudaMemcpy(host_output, output, tensor_size * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_expected, expected, tensor_size * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

    int vector_size = nn->layers[nn->num_layers - 1].output.tensor.tensor_dimensions[1];
    int num_vectors = tensor_size / vector_size;

    DATA_TYPE loss = 0.0f;
    for(int i = 0; i < num_vectors; i++) {
        for(int j = 0; j < vector_size; j++) {
            if(host_expected[i * vector_size + j] > 0.5f) {
                DATA_TYPE p = host_output[i * vector_size + j];
                if(isnan(p) || isinf(p) || p <= 0.0f) {
                    fprintf(stderr, "  INVALID softmax output at vec=%d idx=%d val=%f\n", i, j, p);
                    free(host_output);
                    free(host_expected);
                    return -1.0f;
                }
                loss -= logf(fmaxf(p, 1e-10f));
                break;
            }
        }
    }

    free(host_output);
    free(host_expected);
    return loss / num_vectors;
}

int main() {
    printf("Hello, CUDA!\n");

    int tokens_size = 65;
    int context_length = 128;
    int embedding_size = 128;
    int query_key_size = 16;
    int num_heads = 8;
    int ffn_hidden_size = 512;

    int num_layers = 10;
    Layer* layers = (Layer*)malloc(sizeof(*layers) * num_layers);
    int l = 0;

    create_mlp_layer(&layers[l++], 2, (int[]){context_length, tokens_size}, embedding_size);

    create_layernorm_layer(&layers[l++], 2, (int[]){context_length, embedding_size});

    create_attention_layer(&layers[l++], context_length, embedding_size, query_key_size, num_heads);

    create_layernorm_layer(&layers[l++], 2, (int[]){context_length, embedding_size});

    create_mlp_layer(&layers[l++], 2, (int[]){context_length, embedding_size}, ffn_hidden_size);

    create_gelu_layer(&layers[l++], context_length * ffn_hidden_size);

    create_mlp_layer(&layers[l++], 2, (int[]){context_length, ffn_hidden_size}, embedding_size);

    create_layernorm_layer(&layers[l++], 2, (int[]){context_length, embedding_size});

    create_mlp_layer(&layers[l++], 2, (int[]){context_length, embedding_size}, tokens_size);

    create_softmax_layer(&layers[l++], 2, (int[]){context_length, tokens_size}, 1.0);

    int pos_count = context_length * embedding_size;
    DATA_TYPE* pos_encoding;
    DATA_TYPE* pos_encoding_grads;
    cudaMalloc(&pos_encoding, pos_count * sizeof(DATA_TYPE));
    cudaMalloc(&pos_encoding_grads, pos_count * sizeof(DATA_TYPE));
    cudaMemset(pos_encoding_grads, 0, pos_count * sizeof(DATA_TYPE));

    DATA_TYPE* pos_adam_m;
    DATA_TYPE* pos_adam_v;
    cudaMalloc(&pos_adam_m, pos_count * sizeof(DATA_TYPE));
    cudaMalloc(&pos_adam_v, pos_count * sizeof(DATA_TYPE));
    cudaMemset(pos_adam_m, 0, pos_count * sizeof(DATA_TYPE));
    cudaMemset(pos_adam_v, 0, pos_count * sizeof(DATA_TYPE));

    DATA_TYPE* host_pos = (DATA_TYPE*)malloc(pos_count * sizeof(DATA_TYPE));
    for(int i = 0; i < pos_count; i++) {
        host_pos[i] = normal_sample(0.0f, 0.02f);
    }
    cudaMemcpy(pos_encoding, host_pos, pos_count * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    free(host_pos);

    int attn_idx = 2;
    int ffn_idx = 6;
    NN nn = {
        .num_layers = l,
        .layers = layers,
        .pos_encoding = pos_encoding,
        .pos_encoding_grads = pos_encoding_grads,
        .pos_encoding_adam = {.m = pos_adam_m, .v = pos_adam_v},
        .attn_residual_layer_idx = attn_idx,
        .ffn_residual_layer_idx = ffn_idx,
        .grad_scale = 1.0f / (BATCH_SIZE * context_length),
    };

    create_nn(&nn);
    load_nn(&nn, "model.data");
    printf("NN created with %d layers\n", nn.num_layers);

    DATA_TYPE* dataset;
    char* tokens;
    printf("Loading dataset...\n");
    load_language_dataset("language/tinyshakespeare.txt", DATASET_SIZE, &dataset, &tokens);

    srand(42);

    DATA_TYPE learning_rate = LEARNING_RATE;
    int max_pos = DATASET_SIZE - context_length - 1;

    for(int step = 0; step < NUM_STEPS; step++) {
        zero_grads_nn(&nn);
        DATA_TYPE batch_loss = 0.0f;

        for(int b = 0; b < BATCH_SIZE; b++) {
            int pos = rand() % max_pos;
            call_nn(&nn, dataset + pos * 65, 1);

            DATA_TYPE sample_loss = compute_cross_entropy_loss(&nn, dataset + (pos + 1) * 65);
            if(sample_loss < 0.0f) {
                printf("  Invalid loss, aborting at step %d\n", step);
                return 1;
            }
            batch_loss += sample_loss;

            grad_nn(&nn, dataset + (pos + 1) * 65);
        }

        clip_grads_nn(&nn, 1.0f);
        update_nn(&nn, learning_rate, WEIGHT_DECAY);

        int w0_size = nn.layers[0].input.tensor.tensor_dimensions[1] * nn.layers[0].output.tensor.tensor_dimensions[1];
        if(check_nan(nn.layers[0].layer.mlp_layer.weights, w0_size, "emb_weights")) {
            printf("  NaN detected at step %d\n", step);
            return 1;
        }

        DATA_TYPE avg_batch_loss = batch_loss / BATCH_SIZE;
        printf("step %5d | samples %6d | loss %.4f\n", step + 1, (step + 1) * BATCH_SIZE, avg_batch_loss);

        if((step + 1) % 100 == 0) {
            test_nn(&nn, dataset, tokens, 0);
            save_nn(&nn, "model.data");
        }
    }

    cudaFree(nn.pos_encoding);
    cudaFree(nn.pos_encoding_grads);
    cudaFree(nn.pos_encoding_adam.m);
    cudaFree(nn.pos_encoding_adam.v);
    free(layers);
    return 0;
}
