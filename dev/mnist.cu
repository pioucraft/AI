#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "mnist.h"
#include "utils.h"

int load_mnist_dataset(const char* images_path, const char* labels_path, MNIST_Image** dataset, int num_images) {
    *dataset = (MNIST_Image*)malloc(num_images * sizeof(MNIST_Image));

    FILE* images_file = fopen(images_path, "rb");
    FILE* labels_file = fopen(labels_path, "rb");

    unsigned char buffer[4096];
    int read_bytes = 0;

    int total_bytes_images = 0;
    int total_bytes_labels = 0;
    unsigned char* images_buffer = NULL;
    unsigned char* labels_buffer = NULL;

    while((read_bytes = fread(buffer, sizeof(unsigned char), 4096, images_file)) > 0) {
        total_bytes_images += read_bytes;
        images_buffer = (unsigned char*)realloc(images_buffer, total_bytes_images);
        memcpy(images_buffer + total_bytes_images - read_bytes, buffer, read_bytes);
    }

    while((read_bytes = fread(buffer, sizeof(unsigned char), 4096, labels_file)) > 0) {
        total_bytes_labels += read_bytes;
        labels_buffer = (unsigned char*)realloc(labels_buffer, total_bytes_labels);
        memcpy(labels_buffer + total_bytes_labels - read_bytes, buffer, read_bytes);
    }

    DATA_TYPE* c_pixels = (DATA_TYPE*)malloc(28 * 28 * sizeof(DATA_TYPE));
    DATA_TYPE* c_label = (DATA_TYPE*)malloc(10 * sizeof(DATA_TYPE));

    for(int i = 0; i < num_images; i++) {
        MNIST_Image c_image;

        unsigned char* pixels = images_buffer + 16 + i * 28 * 28;
        for(int j = 0; j < 28; j++) {
            for(int k = 0; k < 28; k++) {
                c_pixels[j * 28 + k] = (DATA_TYPE)((DATA_TYPE)pixels[28 * j + k] / 255.0f);
            }
        }

        int label = labels_buffer[8 + i];
        for(int j = 0; j < 10; j++) {
            c_label[j] = (DATA_TYPE)((j == label) ? 1.0f : -1.0f);
        }

        cudaMalloc(&(c_image.pixels), 28 * 28 * sizeof(DATA_TYPE));
        cudaMalloc(&(c_image.label), 10 * sizeof(DATA_TYPE));
        cudaMemcpy(c_image.pixels, c_pixels, 28 * 28 * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
        cudaMemcpy(c_image.label, c_label, 10 * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
        (*dataset)[i] = c_image;
    }

    free(images_buffer);
    free(labels_buffer);
    free(c_pixels);
    free(c_label);
    checkCudaError();
    printf("Loaded %d images and labels into GPU memory.\n", num_images);

    return 0;
}

int display_nn_output_mnist(NN* nn, DATA_TYPE* label) {
    DATA_TYPE* input_host = (DATA_TYPE*)malloc(28 * 28 * sizeof(DATA_TYPE));
    DATA_TYPE* output_host = (DATA_TYPE*)malloc(10 * sizeof(DATA_TYPE));
    DATA_TYPE* label_host = (DATA_TYPE*)malloc(10 * sizeof(DATA_TYPE));

    cudaMemcpy(input_host, nn->layers[0].input.d1.input, 28 * 28 * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_host, nn->layers[nn->num_layers - 1].output.d1.output, 10 * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(label_host, label, 10 * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

    printf("Input:\n");
    for(int i = 0; i < 28; i++) {
        for(int j = 0; j < 28; j++) {
            if(input_host[i * 28 + j] > 0.3f) {
                printf("X");
            } else {
                printf(" ");
            }        }
        printf("\n");
    }

    printf("Output:\n");
    for(int i = 0; i < 10; i++) {
        printf("For %d: got %f, expected %f\n", i, output_host[i], label_host[i]);
    }

    return 0;
}
