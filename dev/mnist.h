#ifndef MNIST_H
#define MNIST_H

#include "utils.h"

typedef struct MNIST_Image {
    DATA_TYPE* pixels; // size 28*28
    DATA_TYPE* label; // size 10
} MNIST_Image;

int load_mnist_dataset(const char* images_path, const char* labels_path, MNIST_Image** dataset, int num_images);

int display_nn_output_mnist(NN* nn, DATA_TYPE* label);

#endif
