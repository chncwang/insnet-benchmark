#include "cxxopts.hpp"
#include "insnet/insnet.h"
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
#include <queue>
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
using std::priority_queue;
using std::unordered_map;
using std::unordered_set;
using std::unique_ptr;
using std::make_unique;
using std::move;
using std::pair;
using std::make_pair;
using std::default_random_engine;
using std::shuffle;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;
using std::chrono::duration;
using insnet::dtype;
using insnet::Vocab;
using insnet::Graph;
using insnet::Node;
using insnet::Profiler;

constexpr int MODEL_TYPE_TRANSFORMER = 0;

vector<string> filterWordList(const unordered_map <string, int> &table, int k) {
    auto cmp = [&table](const string *a, const string *b) {
        return table.at(*a) > table.at(*b);
    };

    vector<const string *> keys;
    for (const auto &it : table) {
        keys.push_back(&it.first);
    }

    priority_queue<const string *, vector<const string *>, decltype(cmp)> queue(cmp);
    for (const string *key : keys) {
        if (queue.size() < k) {
            queue.push(key);
        } else if (table.at(*queue.top()) < table.at(*key)) {
            queue.pop();
            queue.push(key);
        }
    }

    vector<string> ret;
    while (!queue.empty()) {
        const string *e = queue.top();
        ret.push_back(*e);
        queue.pop();
    }
    cout << fmt::format("ret size:{}", ret.size()) << endl;
    return ret;
}

unordered_map<string, int> calWordFreq(const vector<vector<string>> &sentences) {
    unordered_map<string, int> ret;
    for (const auto &s : sentences) {
        for (const string &w : s) {
            auto it = ret.find(w);
            if (it == ret.end()) {
                ret.insert(make_pair(w, 1));
            } else {
                it->second++;
            }
        }
    }
    return ret;
}

pair<float, float> sentenceLenStat(const vector<vector<string>> &sentences) {
    float sum = 0;
    for (const auto &s : sentences) {
        sum += s.size();
    }
    float mean = sum / sentences.size();

    sum = 0;
    for (const auto &s : sentences) {
        float x = s.size() - mean;
        sum += x * x;
    }
    float sd = std::sqrt(sum / sentences.size());

    return make_pair(mean, sd);
}

int main(int argc, const char *argv[]) {
    Options options("InsNet benchmark");
    options.add_options()
        ("model", "model type where 0 means Transformer and 1 means LSTM",
         cxxopts::value<int>()->default_value("0"))
        ("profile", "whether enable profiler",
         cxxopts::value<bool>()->default_value("false"))
        ("device_id", "device id", cxxopts::value<int>()->default_value("0"))
        ("train", "training set file", cxxopts::value<string>())
        ("post", "post file", cxxopts::value<string>())
        ("response", "response file", cxxopts::value<string>())
        ("batch_size", "batch size", cxxopts::value<int>()->default_value("1"))
        ("dropout", "dropout", cxxopts::value<dtype>()->default_value("0.1"))
        ("lr", "learning rate", cxxopts::value<dtype>()->default_value("0.001"))
        ("layer", "layer", cxxopts::value<int>()->default_value("3"))
        ("head", "head", cxxopts::value<int>()->default_value("8"))
        ("hidden_dim", "hidden dim", cxxopts::value<int>()->default_value("512"))
        ("src_vocab_size", "src vocab size", cxxopts::value<int>()->default_value("10000"))
        ("tgt_vocab_size", "tgt vocab size", cxxopts::value<int>()->default_value("10000"))
        ("cutoff", "cutoff", cxxopts::value<int>()->default_value("0"));

    auto args = options.parse(argc, argv);

    int device_id = args["device_id"].as<int>();
    cout << fmt::format("device_id:{}", device_id) << endl;

#if USE_GPU
    insnet::cuda::initCuda(device_id, 0);
#endif

    string train_pair_file = args["train"].as<string>();

    vector<PostAndResponses> train_post_and_responses = readPostAndResponsesVector(
            train_pair_file, 10000000);
    vector<ConversationPair> train_conversation_pairs = toConversationPairs(
            train_post_and_responses);

    cout << "train size:" << train_conversation_pairs.size() << endl;

    string post_file = args["post"].as<string>();
    vector<vector<string>> post_sentences = readSentences(post_file);
    auto post_stat = sentenceLenStat(post_sentences);
    cout << fmt::format("post mean:{} sd:{}", post_stat.first, post_stat.second) << endl;
    string response_file = args["response"].as<string>();
    vector<vector<string>> response_sentences = readSentences(response_file);
    auto res_stat = sentenceLenStat(response_sentences);
    cout << fmt::format("response mean:{} sd:{}", res_stat.first, res_stat.second) << endl;

    unordered_map<string, int> src_word_count_map = calWordFreq(post_sentences);
    int src_vocab_size = args["src_vocab_size"].as<int>();
    vector<string> word_list = filterWordList(src_word_count_map, src_vocab_size);
    word_list.push_back(insnet::UNKNOWN_WORD);

    Vocab src_vocab;
    src_vocab.init(word_list);

    int tgt_vocab_size = args["tgt_vocab_size"].as<int>();
    unordered_map<string, int> tgt_word_count_map = calWordFreq(response_sentences);
    word_list = filterWordList(tgt_word_count_map, tgt_vocab_size);
    word_list.push_back(insnet::UNKNOWN_WORD);
    word_list.push_back(BEGIN_SYMBOL);
    word_list.push_back(STOP_SYMBOL);

    Vocab tgt_vocab;
    tgt_vocab.init(word_list);

    vector<vector<int>> src_ids, tgt_in_ids, tgt_out_ids;
    for (const auto &s : post_sentences) {
        vector<int> ids;
        ids.reserve(s.size());
        for (const auto &w : s) {
            int id;
            if (src_vocab.find_string(w)) {
                id = src_vocab.from_string(w);
            } else {
                id = src_vocab.from_string(insnet::UNKNOWN_WORD);
            }
            ids.push_back(id);
        }
        src_ids.push_back(move(ids));
    }
    for (const auto &s : response_sentences) {
        {
            vector<int> ids;
            ids.reserve(s.size());
            ids.push_back(tgt_vocab.from_string(BEGIN_SYMBOL));
            for (int i = 0; i < s.size(); ++i) {
                const auto &w = s.at(i);
                int id;
                if (tgt_vocab.find_string(w)) {
                    id = tgt_vocab.from_string(w);
                } else {
                    id = tgt_vocab.from_string(insnet::UNKNOWN_WORD);
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
                if (tgt_vocab.find_string(w)) {
                    id = tgt_vocab.from_string(w);
                } else {
                    id = tgt_vocab.from_string(insnet::UNKNOWN_WORD);
                }
                ids.push_back(id);
            }
            ids.push_back(tgt_vocab.from_string(STOP_SYMBOL));
            tgt_out_ids.push_back(move(ids));
        }
    }

    unique_ptr<ModelParams> params;
    int model_type = args["model"].as<int>();
    int hidden_dim = args["hidden_dim"].as<int>();
    int layer = args["layer"].as<int>();
    if (model_type == 0) {
        int head = args["head"].as<int>();
        params = make_unique<TransformerParams>();
        dynamic_cast<TransformerParams &>(*params).init(src_vocab, tgt_vocab, hidden_dim, layer,
                head);
    }

    dtype lr = args["lr"].as<dtype>();
    cout << fmt::format("lr:{}", lr) << endl;
    insnet::AdamOptimizer optimizer(params->tunableParams(), lr);
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

        Profiler &profiler = Profiler::Ins();
        bool enable_profile = args["profile"].as<bool>();
        cout << fmt::format("enable profile:{}", enable_profile) << endl;

        while (batch_begin != train_conversation_pairs.end()) {
            ++iteration;
            if (iteration == BENCHMARK_BEGIN_ITER) {
                begin_time = high_resolution_clock::now();
            }
            auto batch_it = batch_begin;
            int word_sum = 0;
            int tgt_word_sum = 0;

            Graph graph(insnet::ModelStage::TRAINING, false);
            vector<Node *> probs;
            vector<vector<int>> answers;

            if (iteration == BENCHMARK_BEGIN_ITER) {
                profiler.SetEnabled(enable_profile);
                profiler.BeginEvent("top");
            }
            profiler.BeginEvent("graph building");

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
            profiler.EndEvent();

            graph.forward();
            dtype loss = insnet::NLLLoss(probs, tgt_vocab.size(), answers, 1.0f);
            if (iteration % 100 == 0) {
                cout << fmt::format("loss:{} sentence number:{} ppl:{}", loss, sentence_size,
                        std::exp(loss / tgt_word_sum)) << endl;
            }
            graph.backward();
            profiler.BeginEvent("optimize");
            optimizer.step();
            profiler.EndCudaEvent();

            if ((iteration % 100 == 0 && iteration >= BENCHMARK_BEGIN_ITER) ||
                    word_sum_for_benchmark > 4000000) {
                auto now = high_resolution_clock::now();
                auto elapsed_time = duration_cast<milliseconds>(now -
                        begin_time);
                float word_count_per_sec = 1e3 * word_sum_for_benchmark /
                    static_cast<float>(elapsed_time.count());
                cout << fmt::format("begin:{} now:{}", begin_time.time_since_epoch().count(),
                        now.time_since_epoch().count());
                cout << fmt::format("epoch:{} iteration:{} word_count_per_sec:{} word count:{} time:{} step time:{}",
                        epoch, iteration, word_count_per_sec, word_sum_for_benchmark,
                        elapsed_time.count(),
                        elapsed_time.count() / (iteration + 1 - BENCHMARK_BEGIN_ITER)) << endl;
            }

            batch_begin = batch_it;
            if (word_sum_for_benchmark > 4000000) {
                cout << "benchmark end" << endl;
                profiler.EndEvent();
                profiler.Print();
                exit(0);
            }
        }
    }

    return 0;
}
