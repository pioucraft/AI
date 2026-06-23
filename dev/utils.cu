#include <stdio.h>

#include "utils.h"

void checkCudaError() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error detected\n");
        fprintf(stderr, "  code : %d\n", (int)err);
        fprintf(stderr, "  name : %s\n", cudaGetErrorName(err));
        fprintf(stderr, "  desc : %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
