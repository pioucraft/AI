#include <stdio.h>
#include <math.h>

#define TYPE float
#define NUM_LAYERS 4

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

__device__ void create_nn(NN* nn, int num_layers, int num_neurons_per_layer, int num_in, int num_out) {
    cudaMalloc(&(nn->layers), sizeof(Layer) * num_layers);
    nn->num_layers = num_layers;

    for(int c_layer = 0; c_layer < num_layers; c_layer++) {
        int c_num_neurons = c_layer == num_layers - 1 ? num_out : num_neurons_per_layer;
        cudaMalloc(&(nn->layers[c_layer].neurons),  sizeof(Neuron) * c_num_neurons);
        nn->layers[c_layer].num_neurons = c_num_neurons;
        
        for(int c_neuron = 0; c_neuron < c_num_neurons; c_neuron++) {
            int c_num_weights = c_layer == 0 ? num_in : num_neurons_per_layer;
            cudaMalloc(&(nn->layers[c_layer].neurons[c_neuron].weights), sizeof(TYPE) * c_num_weights);
            nn->layers[c_layer].neurons[c_neuron].num_weights = c_num_weights;
    
            cudaMalloc(&(nn->layers[c_layer].neurons[c_neuron].weights_grads), sizeof(TYPE) * c_num_weights);

            for(int c_weight = 0; c_weight < c_num_weights; c_weight++) {
                TYPE random_weight = ((TYPE)rand() / (TYPE)RAND_MAX * (TYPE)2.0 - (TYPE)1.0);
                random_weight = random_weight * sqrt(8.0/ (TYPE)c_num_weights);

                nn->layers[c_layer].neurons[c_neuron].weights[c_weight] = random_weight;
            }
            TYPE random_bias = ((TYPE)rand() / (TYPE)RAND_MAX * (TYPE)2.0 - (TYPE)1.0);
            random_bias = random_bias * sqrt(8.0/ (TYPE)c_num_weights);

            nn->layers[c_layer].neurons[c_neuron].bias = random_bias;
        }
    }
}

__global__ void main_kernel() {
    NN nn;
    create_nn(&nn, NUM_LAYERS, 128, 28 * 28, 10);
}

int main() {
    main_kernel<<<1, 1>>>();
    return 0;
}
