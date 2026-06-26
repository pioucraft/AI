#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "convolution.h"
#include "dropout.h"
#include "layernorm.h"
#include "mlp.h"
#include "../mnist/mnist.h"
#include "nn.h"
#include "pooling.h"
#include "relu.h"
#include "softmax.h"
#include "tanh.h"
#include "utils.h"

#define NUM_CYCLES 100
#define DATASET_SIZE 60000
#define TEST_DATASET_SIZE 10000
#define BATCH_SIZE 64
#define LEARNING_RATE 5e-2
#define WEIGHT_DECAY 1e-2

int main() {
    printf("Hello, CUDA!\n");

    MNIST_Image* dataset;
    load_mnist_dataset("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", &dataset, DATASET_SIZE);

    MNIST_Image* test_dataset;
    load_mnist_dataset("mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte", &test_dataset, TEST_DATASET_SIZE);

    int num_layers = 14;
    int c_layer = 0;
    Layer* layers = (Layer*)malloc(sizeof(*layers) * num_layers);

    int multiplier = 32;
    create_convolution_layer(&(layers[c_layer++]), 28, 26, 3, 1*multiplier, 1, 1*multiplier);
    create_pooling_layer(&(layers[c_layer++]), 26, 13, 2, 1*multiplier);
    create_relu_layer(&(layers[c_layer++]), 13*13*1*multiplier);
    create_dropout_layer(&(layers[c_layer++]), 13*13*1*multiplier, 0.25f);

    create_convolution_layer(&(layers[c_layer++]), 13, 10, 4, 2*multiplier, 1*multiplier, 2*multiplier);
    create_pooling_layer(&(layers[c_layer++]), 10, 5, 2, 2*multiplier);
    create_relu_layer(&(layers[c_layer++]), 5*5*2*multiplier);
    create_dropout_layer(&(layers[c_layer++]), 5*5*2*multiplier, 0.25f);

    create_mlp_layer(&(layers[c_layer++]), 2, (int[]){1, 5*5*2*multiplier}, 128);

    create_layernorm_layer(&(layers[c_layer++]), 2, (int[]){1, 128});
    create_relu_layer(&(layers[c_layer++]), 128);
    create_dropout_layer(&(layers[c_layer++]), 128, 0.5f);

    create_mlp_layer(&(layers[c_layer++]), 2, (int[]){1, 128}, 10);
    create_softmax_layer(&(layers[c_layer++]), 2, (int[]){1, 10}, 1.0f);
    // create_tanh_layer(&(layers[c_layer++]), 10);

    NN nn = {
        .num_layers = num_layers,
        .layers = layers
    };

    create_nn(&nn);
    // load_nn(&nn, "model.data");

    for(int cycle = 0; cycle < NUM_CYCLES; cycle++) {
        printf("Cycle %d\n", cycle);

        int correct_predictions = 0;
        for(int i = 0; i < TEST_DATASET_SIZE; i++) {
            call_nn(&nn, test_dataset[i].pixels, 0);
            DATA_TYPE output[10];
            cudaMemcpy(output, nn.layers[nn.num_layers - 1].output.tensor.output, 10 * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

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

        DATA_TYPE learning_rate = LEARNING_RATE;

        for(int i = 0; i < DATASET_SIZE - BATCH_SIZE; i += BATCH_SIZE) {
            zero_grads_nn(&nn);
            for(int j = 0; j < BATCH_SIZE; j++) {
                call_nn(&nn, dataset[i + j].pixels, 1);
                grad_nn(&nn, dataset[i + j].label);
                if((i + j) % 10000 == 0) {
                    printf("Processed %d samples\n", i + j);
                }
            }
            clip_grads_nn(&nn, 1.0f);
            update_nn(&nn, learning_rate, WEIGHT_DECAY);
        }
        save_nn(&nn, "model.data");
        
    }

    return 0;
}

