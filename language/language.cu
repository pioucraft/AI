#include <cuda_runtime_api.h>
#include <stdio.h>
#include <string.h>

#include "language.h"

int create_tokenizer(char* file_content, char** tokens, int dataset_size) {
    *tokens = (char*)malloc(1);
    (*tokens)[0] = '\0';
    for(int i = 0; i < dataset_size; i++) {
        char c = file_content[i];
        int found = 0;
        for(int j = 0; j < strlen(*tokens); j++) {
            if(c == (*tokens)[j]) {
                found = 1;
            }
        }
        if(!found) {
            *tokens = (char*)realloc(*tokens, strlen(*tokens) + 2);
            strncat(*tokens, &c, 1);
        }
    }

    printf("%d tokens found: %s\n", (int)strlen(*tokens), *tokens);

    return 0;
}

int tokenizer(char c, char* tokens) {
    for(int i = 0; i < strlen(tokens); i++) {
        if(c == tokens[i]) {
            return i;
        }
    }
    return -1;
}

int untokenizer(int token, char* tokens) {
    if (token < 0 || token >= (int)strlen(tokens)) return '?';
    return tokens[token];
}

int load_language_dataset(const char* dataset_path, int dataset_size, DATA_TYPE** dataset, char** tokens) {
    FILE* file = fopen(dataset_path, "r");
    if(file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", dataset_path);
        return -1;
    }

    char buffer[4096];
    int read_bytes = 0;
    char* file_content = (char*)malloc(1);
    file_content[0] = '\0';
    while((read_bytes = fread(buffer, sizeof(char), 4096, file)) > 0) {
        file_content = (char*)realloc(file_content, read_bytes + strlen(file_content) + 1);
        strncat(file_content, buffer, read_bytes);
    }
    
    for(int i = 0; i < strlen(file_content); i++) {
        if(file_content[i] == ' ') {
            // file_content[i] = '_';
        }
    }

    create_tokenizer(file_content, tokens, dataset_size);

    int* tokenized_data = (int*)malloc(sizeof(int) * dataset_size);
    for(int i = 0; i < dataset_size; i++) {
        tokenized_data[i] = tokenizer(file_content[i], *tokens);
    }

    cudaMalloc(dataset, sizeof(DATA_TYPE) * dataset_size * strlen(*tokens));
    DATA_TYPE* host_dataset = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * dataset_size * strlen(*tokens));
    for(int i = 0; i < dataset_size; i++) {
        for(int j = 0; j < strlen(*tokens); j++) {
            DATA_TYPE value = (tokenized_data[i] == j) ? 1.0f : 0.0f;
            host_dataset[i * strlen(*tokens) + j] = value;
        }
    }
    cudaMemcpy(*dataset, host_dataset, sizeof(DATA_TYPE) * dataset_size * strlen(*tokens), cudaMemcpyHostToDevice);

    return 0;
}

