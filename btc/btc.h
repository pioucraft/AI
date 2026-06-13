#ifndef BTC_H
#define BTC_H

#include "../dev/utils.h"

int load_btc_datapoints(const char* dataset_path, DATA_TYPE** dataset, int num_datapoints);

int display_nn_output_btc(NN* nn, DATA_TYPE* dataset);

#endif
