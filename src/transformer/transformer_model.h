#ifndef N3LDG_PLUS_BENCHMARK_TRANSFORMER_MODEL_H
#define N3LDG_PLUS_BENCHMARK_TRANSFORMER_MODEL_H

#include <vector>
#include "n3ldg-plus/n3ldg-plus.h"
#include "transformer_params.h"

n3ldg_plus::Node *transformerSeq2seq(const std::vector<int> &src_ids,
        const std::vector<int> &tgt_in_ids,
        n3ldg_plus::Graph &graph,
        ModelParams &params,
        n3ldg_plus::dtype dropout) {
    using n3ldg_plus::Node;
    Node *emb = n3ldg_plus::embedding(graph, src_ids, params.embedding.E);
    TransformerParams &transformer_params = dynamic_cast<TransformerParams &>(params);
    Node *enc = n3ldg_plus::transformerEncoder(*emb, src_ids.size(), transformer_params.encoder,
            dropout).back();
    Node *dec_emb = n3ldg_plus::embedding(graph, tgt_in_ids, params.embedding.E);
    Node *dec = n3ldg_plus::transformerDecoder(*enc, src_ids.size(), *dec_emb, tgt_in_ids.size(),
            transformer_params.decoder, dropout).back();
    dec = n3ldg_plus::linear(*dec, params.embedding.E);
    dec = n3ldg_plus::softmax(*dec, tgt_in_ids.size());
    return dec;
}

#endif
