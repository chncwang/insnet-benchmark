#ifndef N3LDG_PLUS_BENCHMARK_PARAMS_H
#define N3LDG_PLUS_BENCHMARK_PARAMS_H

#include "n3ldg-plus/n3ldg-plus.h"

class ModelParams : public n3ldg_plus::TunableParamCollection
#if USE_GPU
, public n3ldg_plus::cuda::TransferableComponents
#endif
{
public:
    void init(const n3ldg_plus::Vocab &vocab, int hidden_dim, int layer) {
        embedding.init(vocab, hidden_dim);
    }

    n3ldg_plus::Embedding<n3ldg_plus::Param> embedding;
};

#endif
