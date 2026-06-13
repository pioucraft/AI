#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>

#include "btc.h"
#include "../dev/utils.h"

int load_btc_datapoints(const char* dataset_path, DATA_TYPE** dataset, int num_datapoints) {
    FILE* file = fopen(dataset_path, "r");
    if(file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", dataset_path);
        return -1;
    }

    cudaMalloc(&dataset, num_datapoints * 5 * sizeof(DATA_TYPE));

    char buffer[4096];
    int read_bytes = 0;
    char* file_content = (char*)malloc(1);
    file_content[0] = '\0';
    while((read_bytes = fread(buffer, sizeof(char), 4096, file)) > 0) {
        file_content = (char*)realloc(file_content, read_bytes + strlen(file_content) + 1);
        strncat(file_content, buffer, read_bytes);
    }

    char* save_line;
    char* line = strtok_r(file_content, "\n", &save_line);
    
    int i = 0;

    printf("Creating neural network...\n");
    while(i < num_datapoints) {
        strtok_r(NULL, "\n", &save_line);
        printf("%d %s\n", i, line);
        char* save_token;
        // Add a + 1 to skip the opening quote, keeps the closing quote but it's not a problem since atof will ignore it
        DATA_TYPE value = atof(strtok_r(line, ",", &save_token) + 1);
        cudaMemcpy(&((*dataset)[i * 5]), &value, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
        for(int j = 1; j < 5; j++) {
            value = atof(strtok_r(NULL, ",", &save_token));
            cudaMemcpy(&((*dataset)[i * 5 + j]), &value, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
        }
        i++;
    }

    free(file_content);
    fclose(file);
    return 0;
}

