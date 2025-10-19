#include <stdio.h>
#include <stdlib.h>

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

int create_nn(NN* nn, int num_layers, int num_neurons_per_layer, int num_in, int num_out) {
    nn->layers = malloc(sizeof(Layer) * num_layers);
    nn->num_layers = num_layers;

    for(int c_layer = 0; c_layer < num_layers; c_layer++) {
        int c_num_neurons = c_layer == num_layers ? num_out : num_neurons_per_layer;
        nn->layers[c_layer].neurons = malloc(sizeof(Neuron) * c_num_neurons);
        nn->layers[c_layer].num_neurons = c_num_neurons;
        
        for(int c_neuron = 0; c_neuron < c_num_neurons; c_neuron++) {
            int c_num_weights = c_layer == 0 ? num_in : num_neurons_per_layer;
            nn->layers[c_layer].neurons[c_neuron].weights = malloc(sizeof(TYPE) * c_num_weights);
            nn->layers[c_layer].neurons[c_neuron].num_weights = c_num_weights;

            for(int c_weight = 0; c_weight < c_num_weights; c_weight++) {
             nn->layers[c_layer].neurons[c_neuron].weights[c_weight] = 
            }
        }
    }
    return 0;
}

int main() {
    NN nn;
    create_nn(&nn, 4, 28, 28*28, 10);
    return 0;
}
