#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "utils.h"

DATA_TYPE normal_sample(DATA_TYPE mean, DATA_TYPE std) {
    DATA_TYPE u1 = (DATA_TYPE)rand() / ((DATA_TYPE)RAND_MAX + 1.0f);
    DATA_TYPE u2 = (DATA_TYPE)rand() / ((DATA_TYPE)RAND_MAX + 1.0f);
    if(u1 < 1e-10f) u1 = 1e-10f;
    return mean + std * sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

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

__global__ void check_nan_kernel(DATA_TYPE* buffer, int size, int* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        DATA_TYPE val = buffer[idx];
        if (isnan(val) || isinf(val)) {
            *result = idx + 1;
        }
    }
}

int check_nan(DATA_TYPE* buffer, int size, const char* name) {
    int* d_result;
    cudaMalloc(&d_result, sizeof(int));
    cudaMemset(d_result, 0, sizeof(int));

    int num_blocks = (size + NUM_THREADS - 1) / NUM_THREADS;
    check_nan_kernel<<<num_blocks, NUM_THREADS>>>(buffer, size, d_result);

    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    if (h_result != 0) {
        DATA_TYPE h_val;
        cudaMemcpy(&h_val, buffer + (h_result - 1), sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
        fprintf(stderr, "  NAN/INF in %s at index %d (value=%f)\n", name, h_result - 1, h_val);
        return 1;
    }
    return 0;
}
