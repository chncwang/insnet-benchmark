#ifndef INSNET_BENCHMARK_PARAMS_H
#define INSNET_BENCHMARK_PARAMS_H

#include "insnet/insnet.h"

class ModelParams : public insnet::TunableParamCollection
#if USE_GPU
, public insnet::cuda::TransferableComponents
#endif
{
public:
    void init(const insnet::Vocab &vocab, int hidden_dim, int layer) {
        embedding.init(vocab, hidden_dim);
    }

    insnet::Embedding<insnet::Param> embedding;
};

#endif
