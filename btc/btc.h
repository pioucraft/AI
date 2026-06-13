#ifndef BTC_H
#define BTC_H

#include "../dev/utils.h"

typedef struct BTC_Datapoint {
    DATA_TYPE open;
    DATA_TYPE high;
    DATA_TYPE low;
    DATA_TYPE close;
    DATA_TYPE volume;
} BTC_Datapoint;

int load_btc_datapoints(const char* dataset_path, BTC_Datapoint** dataset, int num_datapoints);

int display_nn_output_btc(NN* nn, DATA_TYPE* label);

#endif
