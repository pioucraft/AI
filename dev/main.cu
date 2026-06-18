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
#define LEARNING_RATE 5e-3

int test_nn(NN* nn, DATA_TYPE* dataset) {
    DATA_TYPE* input;
    cudaMallocManaged(&input, sizeof(DATA_TYPE) * 128 * 65);
    cudaMemcpy(input, dataset, sizeof(DATA_TYPE) * 32 * 65, cudaMemcpyHostToDevice);
    for(int i = 0; i < 64; i++) {
        call_nn(nn, input + i * 65, 1);
        DATA_TYPE max = -1;
        int predicted_token = 0;

        DATA_TYPE* output = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * 65);
        cudaMemcpy(nn->layers[3].output.d1.output, output, sizeof(DATA_TYPE) * 65, cudaMemcpyDeviceToHost);

        for(int j = 0; j < 65; j++) {
            if(output[j] > max) {
                max = output[j];
                predicted_token = j;
            }
        }
    }
}

int main() {
    printf("Hello, CUDA!\n");

    Layer* layers = (Layer*)malloc(sizeof(*layers) * 4);

    int tokens_size = 65; // Number of unique tokens in the dataset

    create_mlp_layer(&layers[0], 65 * 32, 32 * 16); // 64 context length
    create_relu_layer(&layers[1], 32 * 16);

    create_mlp_layer(&layers[2], 32 * 16, 65);
    create_tanh_layer(&layers[3], 65);


    NN nn = {
        .num_layers = 4,
        .layers = layers
    };

    create_nn(&nn);

    DATA_TYPE* dataset;
    printf("Loading dataset...\n");
    load_language_dataset("language/tinyshakespeare.txt", DATASET_SIZE, &dataset);

    for(int cycle = 0; cycle < NUM_CYCLES; cycle++) {
        printf("Cycle %d\n", cycle);

        DATA_TYPE learning_rate = LEARNING_RATE * (1.0f - (float)cycle / NUM_CYCLES);

        for(int i = 0; i < DATASET_SIZE - 65; i++) { // - 64 for context length and -1 for output
            zero_grads_nn(&nn);
            call_nn(&nn, dataset + i * 65, 1);
            grad_nn(&nn, dataset + (i + 1) * 65);
            if(i % 10000 == 0) {
                printf("Processed %d samples\n", i);
                test_nn(&nn, dataset);
            };
            update_nn(&nn, learning_rate / BATCH_SIZE);
        }
        save_nn(&nn, "model.data");
        
    }

    return 0;
}

