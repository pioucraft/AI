#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "convolution.h"
#include "dropout.h"
#include "mlp.h"
#include "mnist.h"
#include "nn.h"
#include "pooling.h"
#include "relu.h"
#include "tanh.h"
#include "utils.h"

#define NUM_CYCLES 100
#define DATASET_SIZE 60000
#define TEST_DATASET_SIZE 10000
#define BATCH_SIZE 64
#define LEARNING_RATE 5e-2

int main() {
    printf("Hello, CUDA!\n");

    MNIST_Image* dataset;
    load_mnist_dataset("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", &dataset, DATASET_SIZE);

    MNIST_Image* test_dataset;
    load_mnist_dataset("mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte", &test_dataset, TEST_DATASET_SIZE);

    Layer* layers = (Layer*)malloc(sizeof(*layers) * 13);

    int multiplier = 32;
    create_convolution_layer(&(layers[0]), 28, 26, 3, 1*multiplier, 1, 1*multiplier);
    create_pooling_layer(&(layers[1]), 26, 13, 2, 1*multiplier);
    create_relu_layer(&(layers[2]), 13*13*1*multiplier);
    create_dropout_layer(&(layers[3]), 13*13*1*multiplier, 0.25f);

    create_convolution_layer(&(layers[4]), 13, 10, 4, 2*multiplier, 1*multiplier, 2*multiplier);
    create_pooling_layer(&(layers[5]), 10, 5, 2, 2*multiplier);
    create_relu_layer(&(layers[6]), 5*5*2*multiplier);
    create_dropout_layer(&(layers[7]), 5*5*2*multiplier, 0.25f);

    create_mlp_layer(&(layers[8]), 5*5*2*multiplier, 128);
    create_relu_layer(&(layers[9]), 128);
    create_dropout_layer(&(layers[10]), 128, 0.5f);
    // create_relu_layer(&(layers[8]), 128);

    create_mlp_layer(&(layers[11]), 128, 10);
    create_tanh_layer(&(layers[12]), 10);

    NN nn = {
        .num_layers = 13,
        .layers = layers
    };

    create_nn(&nn);

    for(int cycle = 0; cycle < NUM_CYCLES; cycle++) {
        printf("Cycle %d\n", cycle);

        int correct_predictions = 0;
        for(int i = 0; i < TEST_DATASET_SIZE; i++) {
            call_nn(&nn, test_dataset[i].pixels, 0);
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

    return 0;
}

