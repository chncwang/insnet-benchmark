#ifndef N3LDG_PLUS_BENCHMARK_TRANSFORMER_PARAMS_H
#define N3LDG_PLUS_BENCHMARK_TRANSFORMER_PARAMS_H

#include "params.h"

class TransformerParams : public ModelParams {
public:
    void init(const n3ldg_plus::Vocab &vocab, int hidden_dim, int layer) override {
        ModelParams::init(vocab, hidden_dim, layer);

        encoder.init(layer, hidden_dim, 8, 100);
        decoder.init(layer, hidden_dim, 8, 100);
    }

    n3ldg_plus::TransformerEncoderParams encoder;
    n3ldg_plus::TransformerDecoderParams decoder;

#if USE_GPU
    std::vector<n3ldg_plus::cuda::Transferable *> transferablePtrs() override {
        return {&embedding, &encoder, &decoder};
    }
#endif

protected:
    virtual std::vector<n3ldg_plus::TunableParam *> tunableComponents() override {
        return {&embedding, &encoder, &decoder};
    }
};

#endif
