#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "mlp.h"
#include "nn.h"
#include "relu.h"
#include "tanh.h"
#include "utils.h"
#include "../language/language.h"

#define NUM_CYCLES 100
#define DATASET_SIZE 1e6
#define BATCH_SIZE 1
#define LEARNING_RATE 1e-4

int test_unembed(DATA_TYPE* embedded) {
    DATA_TYPE max = -1.0;
    int predicted_token = -1;
    for(int j = 0; j < 65; j++) {
        if(embedded[j] > max) {
            max = embedded[j];
            predicted_token = j;
        }
    }

    return predicted_token;
}

int test_nn(NN* nn, DATA_TYPE* dataset, char* tokens) {
    DATA_TYPE* input;
    cudaMallocManaged(&input, sizeof(DATA_TYPE) * 128 * 65);
    cudaMemcpy(input, dataset, sizeof(DATA_TYPE) * 32 * 65, cudaMemcpyHostToDevice);
    printf("Testing NN...\n");
    for(int i = 0; i < 64; i++) {
        call_nn(nn, input + i * 65, 0);
        
        int predicted_token = test_unembed(nn->layers[3].output.d1.output);
        for(int j = 0; j < 65; j++) {
            input[32 * 65 + i * 65 + j] = predicted_token == j ? 1.0 : -1.0;
        }
        for(int j = 0; j < 32 + i; j++) {
            int current_token = test_unembed(input + j * 65);
            char current_char = untokenizer(current_token, tokens);
            if(j == 32) printf("...");
            printf("%c", current_char);
        }
        printf("\n");
    }
    return 0;
}

int main() {
    printf("Hello, CUDA!\n");

    Layer* layers = (Layer*)malloc(sizeof(*layers) * 4);

    int tokens_size = 65; // Number of unique tokens in the dataset

    create_mlp_layer(&layers[0], 2, (int[]){1, 65 * 32}, 32 * 16); // 32 context length
    create_relu_layer(&layers[1], 32 * 16);

    create_mlp_layer(&layers[2], 2, (int[]){1, 32 * 16}, 65);
    create_tanh_layer(&layers[3], 65);


    NN nn = {
        .num_layers = 4,
        .layers = layers
    };

    create_nn(&nn);
    load_nn(&nn, "model.data");

    DATA_TYPE* dataset;
    char* tokens;
    printf("Loading dataset...\n");
    load_language_dataset("language/tinyshakespeare.txt", DATASET_SIZE, &dataset, &tokens);


    for(int cycle = 0; cycle < NUM_CYCLES; cycle++) {
        printf("Cycle %d\n", cycle);

        DATA_TYPE learning_rate = LEARNING_RATE * (1.0f - (float)cycle / NUM_CYCLES);

        for(int i = 0; i < DATASET_SIZE - 65; i++) { // - 64 for context length and -1 for output
            zero_grads_nn(&nn);
            call_nn(&nn, dataset + i * 65, 1);
            grad_nn(&nn, dataset + (i + 32) * 65);
            if(i % 10000 == 0) {
                test_nn(&nn, dataset, tokens);
                printf("Processed %d samples\n", i);
                save_nn(&nn, "model.data");
            };
            update_nn(&nn, learning_rate / BATCH_SIZE);
        }
        
    }

    return 0;
}

