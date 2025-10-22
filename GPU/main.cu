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

void create_nn(NN* nn, int nin, int nout, int num_layers, int num_neurons_per_layer) {
    int num_neurons = (num_layers - 1) * num_neurons_per_layer + nout;
    int num_weights = nin * num_neurons_per_layer + num_neurons_per_layer * num_neurons_per_layer * (num_layers - 2) + num_neurons_per_layer * nout;
    Neuron* neurons;
    TYPE* weights;
    TYPE* weights_grads;
    cudaMalloc(&neurons, sizeof(Neuron) * num_neurons);
    cudaMalloc(&weights, sizeof(TYPE) * num_weights);
    cudaMalloc(&weights_grads, sizeof(TYPE) * num_weights);
    int c_neuron_total = 0;
    int c_weight_total = 0;

    cudaMalloc(&nn->layers, sizeof(Layer) * num_layers);
    nn->num_layers = num_layers;
    
    for(int c_layer = 0; c_layer < num_layers; c_layer++) {

        Layer layer;
        int c_num_neurons = c_layer == num_layers - 1 ? nout : num_neurons_per_layer;
        layer.num_neurons = c_num_neurons;
        layer.neurons = neurons + c_neuron_total;
        
        for(int c_neuron = 0; c_neuron < c_num_neurons; c_neuron++) {

            Neuron neuron;

            int c_num_weights = c_layer == 0 ? nin : num_neurons_per_layer;
            neuron.num_weights = c_num_weights;
            neuron.weights = weights + c_weight_total;
            neuron.weights_grads = weights_grads + c_weight_total;

            for(int c_weight = 0; c_weight < c_num_weights; c_weight++) {
                TYPE random_value = (TYPE)rand() / (TYPE)RAND_MAX * 2 - 1;
                random_value = (TYPE)(random_value * (sqrt(8.0 / c_num_weights)));
                cudaMemcpy(weights + c_weight_total, &random_value, sizeof(TYPE), cudaMemcpyHostToDevice);
                c_weight_total++;
            }

            TYPE random_value = (TYPE)rand() / (TYPE)RAND_MAX * 2 - 1;
            random_value = random_value * (sqrt(8.0 / c_num_weights));
            neuron.bias = random_value;

            cudaMemcpy(neurons + c_neuron_total, &neuron, sizeof(Neuron), cudaMemcpyHostToDevice);
            c_neuron_total++;
        }
        cudaMemcpy(nn->layers + c_layer, &layer, sizeof(Layer), cudaMemcpyHostToDevice);
    }
}

int main() {
    NN nn;
    create_nn(&nn, 28 * 28, 10, NUM_LAYERS, NUM_NEURONS_PER_LAYER);
    cudaDeviceSynchronize();
    return 0;
}
