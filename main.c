#include <stdio.h>

#define TYPE double

typedef struct Neuron {
    TYPE* weights;
    int num_weights;
    TYPE bias;
} Neuron;

typedef struct Layer {
    Neuron* neurons;
    int num_neurons;
} Layer;

typedef struct NN {
    Layer* layers;
    int num_layers;
} NN;

int create_nn(NN* nn, int num_layers) {
    return 0;
}

int main() {
    NN nn;
    create_nn(&nn);
    return 0;
}
