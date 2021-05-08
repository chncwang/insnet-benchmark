#include "cxxopts.hpp"
#include "n3ldg-plus/n3ldg-plus.h"
#include <unistd.h>
#include <chrono>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <iomanip>
#include <array>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <mutex>
#include <atomic>
#include "conversation_structure.h"
#include "data_manager.h"
#include "params.h"
#include "def.h"
#include "transformer/transformer_params.h"
#include "transformer/transformer_model.h"

using cxxopts::Options;
using std::string;
using std::cout;
using std::endl;
using std::vector;
using std::unordered_map;
using std::unordered_set;
using std::unique_ptr;
using std::make_unique;
using std::move;
using std::default_random_engine;
using std::shuffle;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;
using std::chrono::duration;
using n3ldg_plus::dtype;
using n3ldg_plus::Vocab;
using n3ldg_plus::Graph;
using n3ldg_plus::Node;

constexpr int MODEL_TYPE_TRANSFORMER = 0;

int main(int argc, const char *argv[]) {
    Options options("N3LDG++ benchmark");
    options.add_options()
        ("model", "model type where 0 means Transformer and 1 means LSTM",
         cxxopts::value<int>()->default_value("0"))
        ("device_id", "device id", cxxopts::value<int>()->default_value("0"))
        ("train", "training set file", cxxopts::value<string>())
        ("post", "post file", cxxopts::value<string>())
        ("response", "response file", cxxopts::value<string>())
        ("batch_size", "batch size", cxxopts::value<int>()->default_value("1"))
        ("dropout", "dropout", cxxopts::value<dtype>()->default_value("0.1"))
        ("lr", "learning rate", cxxopts::value<dtype>()->default_value("0.001"))
        ("layer", "layer", cxxopts::value<int>()->default_value("3"))
        ("hidden_dim", "hidden dim", cxxopts::value<int>()->default_value("512"));

    auto args = options.parse(argc, argv);

    int device_id = args["device_id"].as<int>();
    cout << fmt::format("device_id:{}", device_id) << endl;

#if USE_GPU
    n3ldg_plus::cuda::initCuda(device_id, 0);
#endif

    string train_pair_file = args["train"].as<string>();

    vector<PostAndResponses> train_post_and_responses = readPostAndResponsesVector(
            train_pair_file);
    vector<ConversationPair> train_conversation_pairs = toConversationPairs(train_post_and_responses);

    cout << "train size:" << train_conversation_pairs.size() << endl;

    string post_file = args["post"].as<string>();
    vector<vector<string>> post_sentences = readSentences(post_file);
    string response_file = args["response"].as<string>();
    vector<vector<string>> response_sentences = readSentences(response_file);

    vector<vector<string>> all_sentences;
    for (auto &p : train_conversation_pairs) {
        auto &s = response_sentences.at(p.response_id);
        all_sentences.push_back(s);
        auto &s2 = post_sentences.at(p.post_id);
        all_sentences.push_back(s2);
    }

    unordered_map<string, int> word_count_map;
    for (const auto &s : all_sentences) {
        for (const string &w : s) {
            auto it = word_count_map.find(w);
            if (it == word_count_map.end()) {
                word_count_map.insert(make_pair(w, 1));
            } else {
                it->second++;
            }
        }
    }

    vector<string> word_list;
    for (const auto &it : word_count_map) {
        if (it.second >= 1000) {
            word_list.push_back(it.first);
        }
    }
    word_list.push_back(n3ldg_plus::UNKNOWN_WORD);
    word_list.push_back(BEGIN_SYMBOL);
    word_list.push_back(STOP_SYMBOL);

    Vocab vocab;
    vocab.init(word_list);

    vector<vector<int>> src_ids, tgt_in_ids, tgt_out_ids;
    for (const auto &s : post_sentences) {
        vector<int> ids;
        ids.reserve(s.size());
        for (const auto &w : s) {
            int id;
            if (vocab.find_string(w)) {
                id = vocab.from_string(w);
            } else {
                id = vocab.from_string(n3ldg_plus::UNKNOWN_WORD);
            }
            ids.push_back(id);
        }
        src_ids.push_back(move(ids));
    }
    for (const auto &s : response_sentences) {
        {
            vector<int> ids;
            ids.reserve(s.size());
            ids.push_back(vocab.from_string(BEGIN_SYMBOL));
            for (int i = 0; i < s.size(); ++i) {
                const auto &w = s.at(i);
                int id;
                if (vocab.find_string(w)) {
                    id = vocab.from_string(w);
                } else {
                    id = vocab.from_string(n3ldg_plus::UNKNOWN_WORD);
                }
                ids.push_back(id);
            }
            tgt_in_ids.push_back(move(ids));
        } {
            vector<int> ids;
            ids.reserve(s.size());
            for (int i = 0; i < s.size(); ++i) {
                const auto &w = s.at(i);
                int id;
                if (vocab.find_string(w)) {
                    id = vocab.from_string(w);
                } else {
                    id = vocab.from_string(n3ldg_plus::UNKNOWN_WORD);
                }
                ids.push_back(id);
            }
            ids.push_back(vocab.from_string(STOP_SYMBOL));
            tgt_out_ids.push_back(move(ids));
        }
    }

    unique_ptr<ModelParams> params;
    int model_type = args["model"].as<int>();
    if (model_type == 0) {
        params = make_unique<TransformerParams>();
    }
    int hidden_dim = args["hidden_dim"].as<int>();
    int layer = args["layer"].as<int>();
    params->init(vocab, hidden_dim, layer);

    dtype lr = args["lr"].as<dtype>();
    cout << fmt::format("lr:{}", lr) << endl;
    n3ldg_plus::AdamOptimzer optimizer(params->tunableParams(), lr);
    int iteration = -1;
    const int BENCHMARK_BEGIN_ITER = 100;

    for (int epoch = 0; ; ++epoch) {
        default_random_engine engine(0);
        shuffle(train_conversation_pairs.begin(), train_conversation_pairs.end(), engine);

        auto batch_begin = train_conversation_pairs.begin();
        int batch_size = args["batch_size"].as<int>();
        dtype dropout = args["dropout"].as<dtype>();

        decltype(high_resolution_clock::now()) begin_time;
        int word_sum_for_benchmark = 0;

        while (batch_begin != train_conversation_pairs.end()) {
            ++iteration;
            if (iteration == BENCHMARK_BEGIN_ITER) {
                begin_time = high_resolution_clock::now();
            }
            auto batch_it = batch_begin;
            int word_sum = 0;
            int tgt_word_sum = 0;

            Graph graph;
            vector<Node *> probs;
            vector<vector<int>> answers;

            int sentence_size = 0;
            while (word_sum < batch_size && batch_it != train_conversation_pairs.end()) {
                const auto &batch_src_ids = src_ids.at(batch_it->post_id);
                word_sum += batch_src_ids.size();
                const auto &batch_tgt_in_ids = tgt_in_ids.at(batch_it->response_id);
                word_sum += batch_tgt_in_ids.size();
                tgt_word_sum += batch_tgt_in_ids.size();

                if (iteration >= BENCHMARK_BEGIN_ITER) {
                    word_sum_for_benchmark += batch_tgt_in_ids.size() + batch_src_ids.size();
                }

                Node *p;
                if (model_type == MODEL_TYPE_TRANSFORMER) {
                    p = transformerSeq2seq(batch_src_ids, batch_tgt_in_ids, graph, *params,
                            dropout);
                }
                probs.push_back(p);
                answers.push_back(tgt_out_ids.at(batch_it->response_id));
                ++batch_it;
                ++sentence_size;
            }

            graph.forward();
            dtype loss = n3ldg_plus::NLLoss(probs, vocab.size(), answers, 1.0f);
            if (iteration % 100 == 0) {
                cout << fmt::format("loss:{} sentence number:{} ppl:{}", loss, sentence_size,
                        std::exp(loss / tgt_word_sum)) << endl;
            }
            graph.backward();
            optimizer.step();

            if (iteration % 100 == 0 && iteration >= BENCHMARK_BEGIN_ITER) {
                auto now = high_resolution_clock::now();
                auto elapsed_time = duration_cast<milliseconds>(now -
                        begin_time);
                float word_count_per_sec = 1e3 * word_sum_for_benchmark /
                    static_cast<float>(elapsed_time.count());
                cout << fmt::format("epoch:{} iteration:{} word_count_per_sec:{} word count:{} time:{}",
                        epoch, iteration, word_count_per_sec, word_sum_for_benchmark,
                        elapsed_time.count()) << endl;
            }

            batch_begin = batch_it;
            if (word_sum_for_benchmark > 1000000) {
                cout << "benchmakr end" << endl;
                exit(0);
            }
        }
    }

    return 0;
}
