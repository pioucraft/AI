#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "dropout.h"
#include "mlp.h"
#include "nn.h"
#include "relu.h"
#include "tanh.h"
#include "utils.h"
#include "../language/language.h"

#define NUM_CYCLES 100
#define DATASET_SIZE 3000
#define TEST_DATASET_SIZE 10000
#define BATCH_SIZE 64
#define LEARNING_RATE 5e-2

int main() {
    printf("Hello, CUDA!\n");

    Layer* layers = (Layer*)malloc(sizeof(*layers) * 8);

    create_mlp_layer(&layers[0], 24 * 5 * 5, 1024);
    create_relu_layer(&layers[1], 1024);
    create_dropout_layer(&layers[2], 1024, 0.5f);

    create_mlp_layer(&layers[3], 1024, 512);
    create_relu_layer(&layers[4], 512);
    create_dropout_layer(&layers[5], 512, 0.25f);

    create_mlp_layer(&layers[6], 512, 1);
    create_tanh_layer(&layers[7], 1);

    NN nn = {
        .num_layers = 8,
        .layers = layers
    };

    create_nn(&nn);

    load_language_dataset("language/tinyshakespeare.txt", 1e6);

    /*
    for(int cycle = 0; cycle < NUM_CYCLES; cycle++) {
        printf("Cycle %d\n", cycle);

        int correct_predictions = 0;
        for(int i = 0; i < TEST_DATASET_SIZE; i++) {
            call_nn(&nn, test_dataset + i * 5, 0);
            DATA_TYPE output[10];
            cudaMemcpy(output, nn.layers[nn.num_layers - 1].output.d1.output, 10 * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

            DATA_TYPE label[10];
            cudaMemcpy(label, test_dataset[i].label, 10 * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

            int predicted_label = 0;
            DATA_TYPE max_output = output[0];
            int correct_label = 0;
            

            for(int j = 0; j < 10; j++) {
                if(output[j] > max_output) {
                    max_output = output[j];
                    predicted_label = j;
                }
                if(label[j] > 0.0f) {
                    correct_label = j;
                }
            }
            if(predicted_label == correct_label) {
                correct_predictions++;
            }
        }
        printf("Test accuracy: %.2f%%\n", (float)correct_predictions / TEST_DATASET_SIZE * 100.0f);

        FILE* accuracy_file = fopen("test_accuracy.data", "a");
        fprintf(accuracy_file, "cycle %d: %.2f%%\n", cycle, (float)correct_predictions / TEST_DATASET_SIZE * 100.0f);
        fclose(accuracy_file);

        DATA_TYPE learning_rate = LEARNING_RATE * (1.0f - (float)cycle / NUM_CYCLES);

        for(int i = 0; i < DATASET_SIZE - BATCH_SIZE; i += BATCH_SIZE) {
            zero_grads_nn(&nn);
            for(int j = 0; j < BATCH_SIZE; j++) {
                call_nn(&nn, dataset[i + j].pixels, 1);
                grad_nn(&nn, dataset[i + j].label);
                if((i + j) % 10000 == 0) {
                    printf("Processed %d samples\n", i + j);
                }
            }
            update_nn(&nn, learning_rate / BATCH_SIZE);
        }
        save_nn(&nn, "model.data");
        
    }
    */

    return 0;
}

