#ifndef N3LDG_PLUS_BENCHMARK_DATA_MANAGER_H
#define N3LDG_PLUS_BENCHMARK_DATA_MANAGER_H

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <codecvt>
#include <fstream>
#include <iterator>
#include <regex>
#include <iostream>
#include <utility>
#include <atomic>
#include <mutex>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/regex.hpp>
#include "conversation_structure.h"
#include "tinyutf8.h"
#include "def.h"
#include "fmt/core.h"

std::vector<PostAndResponses> readPostAndResponsesVector(const std::string &filename) {
    std::vector<PostAndResponses> results;
    std::string line;
    std::ifstream ifs(filename);
    while (std::getline(ifs, line)) {
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(":"));
        if (strs.size() != 2) {
            abort();
        }
        int post_id = stoi(strs.at(0));
        PostAndResponses post_and_responses;
        post_and_responses.post_id = post_id;
        std::vector<std::string> strs2;
        boost::split(strs2, strs.at(1), boost::is_any_of(","));
        if (strs2.empty()) {
            std::cerr << "readPostAndResponsesVector - no response id found!" << line << std::endl;
            abort();
        }
        for (std::string &str : strs2) {
            post_and_responses.response_ids.push_back(stoi(str));
        }
        results.push_back(std::move(post_and_responses));
    }

    return results;
}

std::vector<ConversationPair> toConversationPairs(const PostAndResponses &post_and_responses) {
    std::vector<ConversationPair> results;
    for (int response_id : post_and_responses.response_ids) {
        ConversationPair conversation_pair(post_and_responses.post_id, response_id);
        results.push_back(std::move(conversation_pair));
    }
    return results;
}

std::vector<ConversationPair> toConversationPairs(
        const std::vector<PostAndResponses> &post_and_responses_vector) {
    std::vector<ConversationPair> results;
    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        std::vector<ConversationPair> conversation_pairs = toConversationPairs(post_and_responses);
        for (const ConversationPair & conversation_pair : conversation_pairs) {
            results.push_back(conversation_pair);
        }
    }
    return results;
}

std::vector<ConversationPair> readConversationPairs(const std::string &filename) {
    std::vector<PostAndResponses> post_and_responses_vector = readPostAndResponsesVector(filename);
    std::vector<ConversationPair> results;
    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        std::vector<ConversationPair> conversation_pairs = toConversationPairs(post_and_responses);
        for (ConversationPair &conversation_pair : conversation_pairs) {
            results.push_back(std::move(conversation_pair));
        }
    }

    return results;
}

std::vector<std::vector<std::string>> readSentences(const std::string &filename) {
    std::string line;
    std::ifstream ifs(filename);
    std::vector<std::vector<std::string>> results;

    int i = 0;
    while (std::getline(ifs, line)) {
        std::vector<std::string> strs;
        boost::split_regex(strs, line, boost::regex("##"));
        int index = stoi(strs.at(0));
        if (i != index) {
            abort();
        }

        const std::string &sentence = strs.at(1);
        std::vector<std::string> words;
        boost::split(words, sentence, boost::is_any_of(" "));
        std::vector<utf8_string> utf8_words;
        for (const std::string &word : words) {
            utf8_string s(word);
            utf8_words.push_back(s);
        }

        std::vector<std::string> characters;
        for (const utf8_string &word : utf8_words) {
            characters.push_back(word.cpp_str());
        }

        results.push_back(characters);
        ++i;
    }

    return results;
}

#endif
