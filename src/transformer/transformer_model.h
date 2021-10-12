#ifndef INSNET_BENCHMARK_TRANSFORMER_MODEL_H
#define INSNET_BENCHMARK_TRANSFORMER_MODEL_H

#include <vector>
#include "insnet/insnet.h"
#include "transformer_params.h"

insnet::Node *transformerSeq2seq(const std::vector<int> &src_ids,
        const std::vector<int> &tgt_in_ids,
        insnet::Graph &graph,
        ModelParams &params,
        insnet::dtype dropout) {
    using insnet::Node;
    Node *emb = insnet::embedding(graph, src_ids, params.embedding.E);
    TransformerParams &transformer_params = dynamic_cast<TransformerParams &>(params);
    Node *enc = insnet::transformerEncoder(*emb, transformer_params.encoder, dropout).back();
    Node *dec_emb = insnet::embedding(graph, tgt_in_ids, params.embedding.E);
    Node *dec = insnet::transformerDecoder(*enc, *dec_emb, transformer_params.decoder,
            dropout).back();
    dec = insnet::linear(*dec, params.embedding.E);
    dec = insnet::softmax(*dec, params.embedding.size());
    return dec;
}

#endif
