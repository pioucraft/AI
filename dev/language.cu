#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>

#include "mlp.h"
#include "nn.h"
#include "relu.h"
#include "utils.h"
#include "attention.h"
#include "layernorm.h"
#include "softmax.h"
#include "../language/language.h"
#include <math.h>

#define NUM_CYCLES 10
#define DATASET_SIZE 1000000
#define BATCH_SIZE 64
#define LEARNING_RATE 3e-4
#define WEIGHT_DECAY 1e-2

int test_unembed(DATA_TYPE* probs) {
    DATA_TYPE min = 1.0;
    DATA_TYPE random = (DATA_TYPE)rand() / RAND_MAX;
    DATA_TYPE cumulative = 0.0;
    int predicted_token = -1;
    for(int j = 0; j < 65; j++) {
        cumulative += probs[j];
        if(cumulative >= random) {
            predicted_token = j;
            break;
        }
    }
    return predicted_token;
}

int test_nn(NN* nn, DATA_TYPE* dataset, char* tokens, int pos) {
    DATA_TYPE* host_context = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * 128 * 65);
    DATA_TYPE* device_context;
    cudaMalloc(&device_context, sizeof(DATA_TYPE) * 128 * 65);

    cudaMemcpy(device_context, dataset + pos * 65, sizeof(DATA_TYPE) * 128 * 65, cudaMemcpyDeviceToDevice);
    cudaMemcpy(host_context, device_context, sizeof(DATA_TYPE) * 128 * 65, cudaMemcpyDeviceToHost);

    char input_text[129];
    for(int j = 0; j < 128; j++) {
        input_text[j] = untokenizer(test_unembed(host_context + j * 65), tokens);
    }
    input_text[128] = '\0';

    char output_text[65];
    for(int step = 0; step < 64; step++) {
        call_nn(nn, device_context, 0);

        cudaMemcpy(host_context + 127 * 65,
                   nn->layers[8].output.tensor.output + 127 * 65,
                   sizeof(DATA_TYPE) * 65,
                   cudaMemcpyDeviceToHost);

        int predicted_token = test_unembed(host_context + 127 * 65);
        output_text[step] = untokenizer(predicted_token, tokens);

        memmove(host_context, host_context + 65, sizeof(DATA_TYPE) * 127 * 65);

        for(int j = 0; j < 65; j++) {
            host_context[127 * 65 + j] = predicted_token == j ? 1.0 : 0.0;
        }

        cudaMemcpy(device_context, host_context, sizeof(DATA_TYPE) * 128 * 65, cudaMemcpyHostToDevice);
    }
    output_text[64] = '\0';

    printf("\x1b[31m----------------------------\x1b[0m\n");
    printf("input:\n%s\n\n", input_text);
    printf("output:\n%s\n", output_text);
    printf("\x1b[31m----------------------------\x1b[0m\n");

    free(host_context);
    cudaFree(device_context);
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
                loss -= logf(fmaxf(host_output[i * vector_size + j], 1e-10f));
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

    int num_layers = 9;
    Layer* layers = (Layer*)malloc(sizeof(*layers) * num_layers);
    int l = 0;

    create_mlp_layer(&layers[l++], 2, (int[]){context_length, tokens_size}, embedding_size);

    create_attention_layer(&layers[l++], context_length, embedding_size, query_key_size, num_heads);

    create_layernorm_layer(&layers[l++], 2, (int[]){context_length, embedding_size});

    create_mlp_layer(&layers[l++], 2, (int[]){context_length, embedding_size}, ffn_hidden_size);

    create_relu_layer(&layers[l++], context_length * ffn_hidden_size);

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
    DATA_TYPE deviation = sqrt(2.0f / embedding_size);
    for(int i = 0; i < pos_count; i++) {
        host_pos[i] = (DATA_TYPE)rand() / RAND_MAX * deviation * 2 - deviation;
    }
    cudaMemcpy(pos_encoding, host_pos, pos_count * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    free(host_pos);

    NN nn = {
        .num_layers = l,
        .layers = layers,
        .pos_encoding = pos_encoding,
        .pos_encoding_grads = pos_encoding_grads,
        .pos_encoding_adam = {.m = pos_adam_m, .v = pos_adam_v}
    };

    create_nn(&nn);
    printf("NN created with %d layers\n", nn.num_layers);

    DATA_TYPE* dataset;
    char* tokens;
    printf("Loading dataset...\n");
    load_language_dataset("language/tinyshakespeare.txt", DATASET_SIZE, &dataset, &tokens);

    for(int cycle = 0; cycle < NUM_CYCLES; cycle++) {
        printf("Cycle %d\n", cycle);

        DATA_TYPE learning_rate = LEARNING_RATE;

        zero_grads_nn(&nn);
        for(int i = 0; i < DATASET_SIZE - context_length - 1; i++) {
            call_nn(&nn, dataset + i * 65, 1);
            grad_nn(&nn, dataset + (i + 1) * 65);
            if((i + 1) % BATCH_SIZE == 0) {
                clip_grads_nn(&nn, 1.0f);
                update_nn(&nn, learning_rate, WEIGHT_DECAY);
                zero_grads_nn(&nn);
            }
            if(i % 100 == 0) {
                DATA_TYPE loss = compute_cross_entropy_loss(&nn, dataset + (i + 1) * 65);
                printf("Loss: %f\n", loss);
                test_nn(&nn, dataset, tokens, 0);
                save_nn(&nn, "model.data");
                printf("Processed %d samples\n", i);
            }
        }
    }

    cudaFree(nn.pos_encoding);
    cudaFree(nn.pos_encoding_grads);
    cudaFree(nn.pos_encoding_adam.m);
    cudaFree(nn.pos_encoding_adam.v);
    free(layers);
    return 0;
}
