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
    
            nn->layers[c_layer].neurons[c_neuron].weights_grads = malloc(sizeof(TYPE) * c_num_weights);

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
            } else if(current_value < 0) { // basically leaky ReLU
                current_value = 0.01 * current_value;
            }
            nn->layers[c_layer].neurons[c_neuron].value = current_value;
        }
    }
}

int grad_nn(NN* nn, TYPE* inputs, TYPE* outputs) {
    for(int c_layer = nn->num_layers - 1; c_layer >= 0; c_layer--) {
        if(c_layer != 0) {
            for(int i = 0; i < nn->layers[c_layer - 1].num_neurons; i++) {
                nn->layers[c_layer - 1].neurons[i].sum_grads = 0;
            }
        }
        for(int c_neuron = 0; c_neuron < nn->layers[c_layer].num_neurons; c_neuron++) {
            if(c_layer == nn->num_layers -1) {
                TYPE error = nn->layers[c_layer].neurons[c_neuron].value - outputs[c_neuron]; 
                nn->layers[c_layer].neurons[c_neuron].grad = error * (1 - nn->layers[c_layer].neurons[c_neuron].value * nn->layers[c_layer].neurons[c_neuron].value);
            }
            else {
                nn->layers[c_layer].neurons[c_neuron].grad = (nn->layers[c_layer].neurons[c_neuron].value > 0 ? 1.0 : 0.01) * nn->layers[c_layer].neurons[c_neuron].sum_grads;
            }

            nn->layers[c_layer].neurons[c_neuron].bias_grad += nn->layers[c_layer].neurons[c_neuron].grad;

            for(int c_weight = 0; c_weight < nn->layers[c_layer].neurons[c_neuron].num_weights; c_weight++) {
                if(c_layer == 0) {
                    nn->layers[c_layer].neurons[c_neuron].weights_grads[c_weight] += nn->layers[c_layer].neurons[c_neuron].grad * inputs[c_weight];
                } else {
                    nn->layers[c_layer].neurons[c_neuron].weights_grads[c_weight] += nn->layers[c_layer].neurons[c_neuron].grad * nn->layers[c_layer - 1].neurons[c_weight].value;
                }
            }
            if(c_layer != 0) {
                for(int i = 0; i < nn->layers[c_layer - 1].num_neurons; i++) {
                    nn->layers[c_layer - 1].neurons[i].sum_grads += nn->layers[c_layer].neurons[c_neuron].weights[i] * nn->layers[c_layer].neurons[c_neuron].grad;
                }
            }
        }
    }
    return 0;
}

void zero_grad_nn(NN* nn) {
    for (int l = 0; l < nn->num_layers; l++) {
        Layer* layer = &nn->layers[l];
        for (int n = 0; n < layer->num_neurons; n++) {
            Neuron* neuron = &layer->neurons[n];
            neuron->bias_grad = 0;
            neuron->sum_grads = 0;
            neuron->grad = 0;

            for (int w = 0; w < neuron->num_weights; w++) {
                neuron->weights_grads[w] = 0;
            }
        }
    }
}

void update_nn(NN* nn, TYPE learning_rate) {
    for (int l = 0; l < nn->num_layers; l++) {
        Layer* layer = &nn->layers[l];
        for (int n = 0; n < layer->num_neurons; n++) {
            Neuron* neuron = &layer->neurons[n];

            // Update weights
            for (int w = 0; w < neuron->num_weights; w++) {
                neuron->weights[w] -= learning_rate * neuron->weights_grads[w];
            }

            // Update bias
            neuron->bias -= learning_rate * neuron->bias_grad;
        }
    }
}

void print_nn_stats(NN* nn, int layer_limit) {
    for (int l = 0; l < nn->num_layers && l < layer_limit; l++) {
        Layer* layer = &nn->layers[l];
        printf("Layer %d: %d neurons\n", l, layer->num_neurons);

        for (int n = 0; n < layer->num_neurons; n++) {
            Neuron* neuron = &layer->neurons[n];

            TYPE weight_sum = 0.0;
            TYPE weight_max = neuron->weights[0];
            TYPE weight_min = neuron->weights[0];
            TYPE grad_sum = 0.0;
            TYPE grad_max = neuron->grad;
            TYPE grad_min = neuron->grad;

            for (int w = 0; w < neuron->num_weights; w++) {
                TYPE val = neuron->weights[w];
                weight_sum += val;
                if (val > weight_max) weight_max = val;
                if (val < weight_min) weight_min = val;

                TYPE g = neuron->weights_grads[w];
                grad_sum += g;
                if (g > grad_max) grad_max = g;
                if (g < grad_min) grad_min = g;
            }

            printf(" Neuron %d: bias=%.4f grad=%.4f | weights sum=%.4f, min=%.4f, max=%.4f | grads sum=%.4f, min=%.4f, max=%.4f\n",
                   n,
                   neuron->bias,
                   neuron->grad,
                   weight_sum,
                   weight_min,
                   weight_max,
                   grad_sum,
                   grad_min,
                   grad_max
            );
        }
        printf("\n");
    }
}

int main() {
srand(42);
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
    
    TYPE* images = malloc(sizeof(TYPE) * (total_bytes - 16));
    for(int i = 0; i < total_bytes - 16; i++) {
        images[i] = (TYPE)imagesS[16 + i] / 255.0;
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
    TYPE* c_image = images;
    
    for(int cycle = 0; cycle < CYCLES; cycle++) {
        TYPE total_loss = 0;
        zero_grad_nn(&nn);
        for(int i = 0; i < DATASET_SIZE; i++) {
            call_nn(&nn, c_image);

            TYPE* outputs = malloc(sizeof(TYPE) * 10);
            for(int output_i = 0; output_i < 10; output_i++) {
                TYPE expected_output = output_i == *c_label ? 1.0 : -1.0;
                outputs[output_i] = expected_output;
                TYPE c_loss = expected_output - nn.layers[nn.num_layers - 1].neurons[output_i].value;
                c_loss = c_loss * c_loss;
                total_loss += c_loss;
            }

            grad_nn(&nn, c_image, outputs);

            c_label++;
            c_image += 28 * 28;
        }
        TYPE average_loss = total_loss / DATASET_SIZE;
        printf("%f\n", average_loss);
        // print_nn_stats(&nn, 3);
        c_label = labels;
        c_image = images;
        update_nn(&nn, LEARNING_RATE);
    }

    return 0;
}
