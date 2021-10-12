#ifndef INSNET_BENCHMARK_TRANSFORMER_PARAMS_H
#define INSNET_BENCHMARK_TRANSFORMER_PARAMS_H

#include "params.h"

class TransformerParams : public ModelParams {
public:
    void init(const insnet::Vocab &vocab, int hidden_dim, int layer, int head) {
        ModelParams::init(vocab, hidden_dim, layer);

        encoder.init(layer, hidden_dim, head, 100);
        decoder.init(layer, hidden_dim, head, 100);
    }

    insnet::TransformerEncoderParams encoder;
    insnet::TransformerDecoderParams decoder;

#if USE_GPU
    std::vector<insnet::cuda::Transferable *> transferablePtrs() override {
        return {&embedding, &encoder, &decoder};
    }
#endif

protected:
    virtual std::vector<insnet::TunableParam *> tunableComponents() override {
        return {&embedding, &encoder, &decoder};
    }
};

#endif
