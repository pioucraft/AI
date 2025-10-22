#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define TYPE float
#define NUM_LAYERS 4
#define NUM_NEURONS_PER_LAYER 128
#define CYCLES 10
#define DATASET_SIZE 600
#define BATCH_SIZE 32
#define LEARNING_RATE 1e-3

#define BUFFER_SIZE 1024

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
    cudaDeviceSynchronize();
    
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
    cudaDeviceSynchronize();
}

__global__ void zero_grad(NN nn) {
    for(int l = 0; l < nn.num_layers; l++) {
        Layer* layer = &nn.layers[l];
        if(blockIdx.x < layer->num_neurons) {
            Neuron* neuron = &layer->neurons[blockIdx.x];
            if(threadIdx.x < neuron->num_weights) {
                if(threadIdx.x == 0) {
                    neuron->bias_grad = 0;
                    neuron->sum_grads = 0;
                    neuron->grad = 0;
                }
                neuron->weights_grads[threadIdx.x] = 0;
            }
        }
    }
}

__global__ void call_layer(NN nn, int c_layer, TYPE* inputs) {

    Layer* layer = &nn.layers[c_layer];
    
    if(blockIdx.x < layer->num_neurons) {
        Neuron* neuron = &layer->neurons[blockIdx.x];
        if(threadIdx.x < neuron->num_weights) {
            __shared__ TYPE partials[1024];
            __syncthreads();
            if(c_layer == 0) {
                partials[threadIdx.x] = neuron->weights[threadIdx.x] * inputs[threadIdx.x];
            } else {
                partials[threadIdx.x] = neuron->weights[threadIdx.x] * nn.layers[c_layer - 1].neurons[threadIdx.x].value;
            }
            __syncthreads();


            int finished = 0;
            int divisibleBy = 2;
            int total_finished = 0;

            while(!total_finished) {
                if(!finished && threadIdx.x % divisibleBy == 0) {
                    if(threadIdx.x + divisibleBy / 2 < neuron->num_weights) {
                        partials[threadIdx.x] += partials[threadIdx.x + divisibleBy / 2];
                    }
                } else {
                    finished = 1;
                }
                if(divisibleBy > 512) {
                    total_finished = 1;
                }
                divisibleBy *= 2;
                __syncthreads();
            }

            if(threadIdx.x == 0) {
                neuron->value = partials[0] + neuron->bias;
                if(c_layer == nn.num_layers - 1) {
                    neuron->value = tanh(neuron->value);
                } else if(neuron->value < 0) {
                    neuron->value = neuron->value * 0.01;
                }
            }
        }
    }
}

void call_nn(NN nn, TYPE* inputs) {
    for(int i = 0; i < NUM_LAYERS; i++) {
        call_layer<<<NUM_NEURONS_PER_LAYER, 28 * 28>>>(nn, i, inputs);
        cudaDeviceSynchronize();
    }
}

__global__ void grad_layer(NN nn, int c_layer, TYPE* inputs, TYPE* outputs, int test) {
    Layer* layer = &nn.layers[c_layer];
    if(blockIdx.x < layer->num_neurons) {
        Neuron* neuron = &layer->neurons[blockIdx.x];
        if(threadIdx.x < neuron->num_weights) {
            if(threadIdx.x == 0) {
                if(c_layer == nn.num_layers - 1) {
                    TYPE error = neuron->value - outputs[blockIdx.x];
                    neuron->grad = 2 * error * (1 - neuron->value * neuron->value);
                } else {
                    TYPE sum_grad = 0.0;
                    for(int i = 0; i < nn.layers[c_layer + 1].num_neurons; i++) {
                        sum_grad += nn.layers[c_layer + 1].neurons[i].weights[blockIdx.x] * nn.layers[c_layer + 1].neurons[i].grad;
                    }
                    neuron->grad = (neuron->value > 0 ? 1 : 0.01) * sum_grad;
                }

                neuron->bias_grad += neuron->grad;
            }
            __syncthreads();

            if(c_layer == 0) {
                neuron->weights_grads[threadIdx.x] += neuron->grad * inputs[threadIdx.x];
            } else {
                neuron->weights_grads[threadIdx.x] += neuron->grad * nn.layers[c_layer - 1].neurons[threadIdx.x].value;
            }

        }
    }
}

void grad_nn(NN nn, TYPE* inputs, TYPE* outputs, int test) {
    for(int i = NUM_LAYERS - 1; i >= 0; i--) {
        grad_layer<<<NUM_NEURONS_PER_LAYER, NUM_NEURONS_PER_LAYER>>>(nn, i, inputs, outputs, test);
        cudaDeviceSynchronize();
    }
}

__global__ void update_nn(NN nn, TYPE learning_rate) {
    for (int l = 0; l < nn.num_layers; l++) {
        Layer* layer = &nn.layers[l];
        if(blockIdx.x < layer->num_neurons) {
            Neuron* neuron = &layer->neurons[blockIdx.x];

            if(threadIdx.x < neuron->num_weights) {
                neuron->weights[threadIdx.x] -= learning_rate * neuron->weights_grads[threadIdx.x];

                if(threadIdx.x == 0) {
                    neuron->bias -= learning_rate * neuron->bias_grad;
                }
            }

        }
    }
}

int main() {
    unsigned char buffer[BUFFER_SIZE];

    FILE *imagesF = fopen("train-images", "rb");
    unsigned char* imagesS = NULL;
    
    int read_bytes = 0;
    int total_bytes = 0;
    while((read_bytes = fread(buffer, sizeof(unsigned char), 256, imagesF)) != 0) {
        total_bytes += read_bytes; 
        imagesS = (unsigned char*) realloc(imagesS, sizeof(unsigned char) * total_bytes);
        memcpy(imagesS + total_bytes - read_bytes, buffer, read_bytes);
    }
    
    TYPE* images = (TYPE*) malloc(sizeof(TYPE) * (total_bytes - 16));
    for(int i = 0; i < total_bytes - 16; i++) {
        images[i] = (TYPE)imagesS[16 + i] / 255.0;
    }

    FILE *labelsF = fopen("train-labels", "rb");
    unsigned char* labelsS = NULL;
    
    read_bytes = 0;
    total_bytes = 0;
    while((read_bytes = fread(buffer, sizeof(unsigned char), BUFFER_SIZE, labelsF)) != 0) {
        total_bytes += read_bytes; 
        labelsS = (unsigned char*) realloc(labelsS, sizeof(unsigned char) * total_bytes);
        memcpy(labelsS + total_bytes - read_bytes, buffer, read_bytes);
    }
    
    TYPE* labels = (TYPE*) malloc(sizeof(TYPE) * (total_bytes - 8) * 10);
    for(int i = 0; i < total_bytes - 8; i++) {
        for(int j = 0; j < 10; j++) {
            if (j == labelsS[i + 8]) {
                labels[i*10 + j] = 1.0;
            } else {
                labels[i*10 + j] = -1.0;
            }
        }
    }

    TYPE* device_labels;
    TYPE* device_images;
    cudaMalloc(&device_labels, sizeof(TYPE) * DATASET_SIZE * 10);
    cudaMalloc(&device_images, sizeof(TYPE) * DATASET_SIZE * 28 * 28);
    cudaMemcpy(device_labels, labels, sizeof(TYPE) * DATASET_SIZE * 10, cudaMemcpyHostToDevice);
    cudaMemcpy(device_images, images, sizeof(TYPE) * DATASET_SIZE * 28 * 28, cudaMemcpyHostToDevice);

    NN nn;
    create_nn(&nn, 28 * 28, 10, NUM_LAYERS, NUM_NEURONS_PER_LAYER);

    TYPE* c_label = device_labels;
    TYPE* c_image = device_images;

    for(int cycle = 0; cycle < CYCLES; cycle++) {
        printf("%d\n", cycle);
        for(int batch_start = 0; batch_start < DATASET_SIZE; batch_start += BATCH_SIZE) {
            zero_grad<<<NUM_NEURONS_PER_LAYER, NUM_NEURONS_PER_LAYER>>>(nn);
            cudaDeviceSynchronize();
            for(int i = batch_start; i < batch_start + BATCH_SIZE && i < DATASET_SIZE; i++) {
                call_nn(nn, c_image);
                grad_nn(nn, c_image, c_label, i == 0);
                c_label += 10;
                c_image += 28 * 28;
            }
            update_nn<<<NUM_NEURONS_PER_LAYER, NUM_NEURONS_PER_LAYER>>>(nn, LEARNING_RATE);
            cudaDeviceSynchronize();
        }
    }

    return 0;
}
