#ifndef INSNET_BENCHMARK_PARAMS_H
#define INSNET_BENCHMARK_PARAMS_H

#include "insnet/insnet.h"

class ModelParams : public insnet::TunableParamCollection
#if USE_GPU
, public insnet::cuda::TransferableComponents
#endif
{
public:
    void init(const insnet::Vocab &src_vocab, const insnet::Vocab &tgt_vocab, int hidden_dim,
            int layer) {
        src_embedding.init(src_vocab, hidden_dim);
        tgt_embedding.init(tgt_vocab, hidden_dim);
    }

    insnet::Embedding<insnet::Param> src_embedding;
    insnet::Embedding<insnet::Param> tgt_embedding;
};

#endif
