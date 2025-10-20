#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>

#define TYPE double
#define DATASET_SIZE 6000
#define CYCLES 100
#define LEARNING_RATE 1e-3

typedef struct Neuron {
    TYPE* weights;
    int num_weights;
    TYPE bias;
    
    TYPE value;
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
        int c_num_neurons = c_layer == num_layers - 1 ? num_out : num_neurons_per_layer;
        nn->layers[c_layer].neurons = malloc(sizeof(Neuron) * c_num_neurons);
        nn->layers[c_layer].num_neurons = c_num_neurons;
        
        for(int c_neuron = 0; c_neuron < c_num_neurons; c_neuron++) {
            int c_num_weights = c_layer == 0 ? num_in : num_neurons_per_layer;
            nn->layers[c_layer].neurons[c_neuron].weights = malloc(sizeof(TYPE) * c_num_weights);
            nn->layers[c_layer].neurons[c_neuron].num_weights = c_num_weights;

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
    return 0;
}

int call_nn(NN* nn, TYPE* inputs) {
    int num_layers = nn->num_layers;
    for(int c_layer = 0; c_layer < num_layers; c_layer++) {
        for(int c_neuron = 0; c_neuron < nn->layers[c_layer].num_neurons; c_neuron++) {

            TYPE current_value = 0;
            for(int c_weight = 0; c_weight < nn->layers[c_layer].neurons[c_neuron].num_weights; c_weight++) {
                current_value += nn->layers[c_layer].neurons[c_neuron].weights[c_weight] *
                    (c_layer == 0 ? inputs[c_weight] : nn->layers[c_layer - 1].neurons[c_weight].value);
            }
            current_value += nn->layers[c_layer].neurons[c_neuron].bias;

            if(c_layer == num_layers - 1) {
                current_value = tanh(current_value);
            } else if(current_value < 0) { // basically ReLU
                current_value = 0;
            }
            nn->layers[c_layer].neurons[c_neuron].value = current_value;
        }
    }
}

int main() {
    unsigned char buffer[256];

    FILE *imagesF = fopen("train-images", "rb");
    unsigned char* imagesS = NULL;
    
    int read_bytes = 0;
    int total_bytes = 0;
    while((read_bytes = fread(buffer, sizeof(unsigned char), 256, imagesF)) != 0) {
        total_bytes += read_bytes; 
        imagesS = realloc(imagesS, sizeof(unsigned char) * total_bytes);
        memcpy(imagesS + total_bytes - read_bytes, buffer, read_bytes);
    }
    
    TYPE* images = malloc(sizeof(TYPE) * total_bytes);
    for(int i = 16; i < total_bytes ; i++) {
        images[i] = (TYPE)imagesS[i] / 255.0;
    }

    FILE *labelsF = fopen("train-labels", "rb");
    unsigned char* labelsS = NULL;
    
    read_bytes = 0;
    total_bytes = 0;
    while((read_bytes = fread(buffer, sizeof(unsigned char), 256, labelsF)) != 0) {
        total_bytes += read_bytes; 
        labelsS = realloc(labelsS, sizeof(unsigned char) * total_bytes);
        memcpy(labelsS + total_bytes - read_bytes, buffer, read_bytes);
    }
    
    unsigned char* labels = labelsS + 8;

    NN nn;
    create_nn(&nn, 4, 10, 28 * 28, 10);
    
    unsigned char* c_label = labels;
    unsigned char* c_image = images;
    
    for(int cycle = 0; cycle < CYCLES) {
        for(int i = 0; i < DATASET_SIZE; i++) {
            
        }
    }

    return 0;
}
