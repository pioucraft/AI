#ifndef LANGUAGE_H
#define LANGUAGE_H

#include "../dev/utils.h"

int load_language_dataset(const char* dataset_path, int dataset_size, DATA_TYPE** dataset, char** tokens);

int untokenizer(int token, char* tokens);

#endif
