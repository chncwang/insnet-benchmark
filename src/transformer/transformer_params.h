#ifndef INSNET_BENCHMARK_TRANSFORMER_PARAMS_H
#define INSNET_BENCHMARK_TRANSFORMER_PARAMS_H

#include "params.h"

class TransformerParams : public ModelParams {
public:
    void init(const insnet::Vocab &src_vocab, const insnet::Vocab &tgt_vocab, int hidden_dim,
            int layer, int head) {
        ModelParams::init(src_vocab, tgt_vocab, hidden_dim, layer);

        encoder.init(layer, hidden_dim, head, 1024);
        decoder.init(layer, hidden_dim, head, 1024);
    }

    insnet::TransformerEncoderParams encoder;
    insnet::TransformerDecoderParams decoder;

#if USE_GPU
    std::vector<insnet::cuda::Transferable *> transferablePtrs() override {
        return {&src_embedding, &tgt_embedding, &encoder, &decoder};
    }
#endif

protected:
    virtual std::vector<insnet::TunableParam *> tunableComponents() override {
        return {&src_embedding, &tgt_embedding, &encoder, &decoder};
    }
};

#endif
