#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define TYPE float
#define NUM_LAYERS 4
#define NUM_NEURONS_PER_LAYER 128

typedef struct Neuron {
    TYPE* weights;
    int num_weights;
    TYPE bias;
    
    TYPE value;

    TYPE grad;
    TYPE* weights_grads;
    TYPE sum_grads;
    TYPE bias_grad;
} Neuron;

typedef struct Layer {
    Neuron* neurons;
    int num_neurons;
} Layer;

typedef struct NN {
    Layer* layers;
    int num_layers;
} NN;

int main() {
    return 0;
}
