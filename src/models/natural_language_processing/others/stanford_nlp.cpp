#include "include/models/natural_language_processing/others/stanford_nlp.h"


using namespace std;



#include <torch/torch.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <iostream>
#include <map>
#include <algorithm>
#include <cmath>

// Vocabulary class for words and tags
class Vocabulary {
public:
    std::map<std::string, int> word_to_id;
    std::map<int, std::string> id_to_word;
    std::map<std::string, int> tag_to_id;
    std::map<int, std::string> id_to_tag;
    int word_vocab_size = 0;
    int tag_vocab_size = 0;
    const int word_unk_id = 0; // <unk>
    const int word_pad_id = 1; // <pad>
    const int tag_pad_id = 0;  // <pad>

    Vocabulary() {
        word_to_id["<unk>"] = word_unk_id;
        id_to_word[word_unk_id] = "<unk>";
        word_to_id["<pad>"] = word_pad_id;
        id_to_word[word_pad_id] = "<pad>";
        tag_to_id["<pad>"] = tag_pad_id;
        id_to_tag[tag_pad_id] = "<pad>";
        word_vocab_size = 2;
        tag_vocab_size = 1;
    }

    void build(const std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>>& data) {
        std::map<std::string, int> word_counts;
        std::map<std::string, int> tag_counts;
        for (const auto& [words, tags] : data) {
            for (const auto& word : words) {
                word_counts[word]++;
            }
            for (const auto& tag : tags) {
                tag_counts[tag]++;
            }
        }
        for (const auto& pair : word_counts) {
            if (pair.second >= 2) { // Minimum count for small dataset
                word_to_id[pair.first] = word_vocab_size;
                id_to_word[word_vocab_size] = pair.first;
                word_vocab_size++;
            }
        }
        for (const auto& pair : tag_counts) {
            tag_to_id[pair.first] = tag_vocab_size;
            id_to_tag[tag_vocab_size] = pair.first;
            tag_vocab_size++;
        }
    }

    int get_word_id(const std::string& word) const {
        auto it = word_to_id.find(word);
        return it != word_to_id.end() ? it->second : word_unk_id;
    }

    int get_tag_id(const std::string& tag) const {
        auto it = tag_to_id.find(tag);
        return it != tag_to_id.end() ? it->second : tag_pad_id;
    }
};

// Character Vocabulary for character-level embeddings
class CharVocabulary {
public:
    std::map<char, int> char_to_id;
    std::map<int, char> id_to_char;
    int char_vocab_size = 0;
    const int char_unk_id = 0; // <unk>
    const int char_pad_id = 1; // <pad>

    CharVocabulary() {
        char_to_id['<unk>'] = char_unk_id;
        id_to_char[char_unk_id] = '<unk>';
        char_to_id['<pad>'] = char_pad_id;
        id_to_char[char_pad_id] = '<pad>';
        char_vocab_size = 2;
    }

    void build(const std::vector<std::string>& sentences) {
        std::set<char> chars;
        for (const auto& sentence : sentences) {
            for (char c : sentence) {
                chars.insert(c);
            }
        }
        for (char c : chars) {
            char_to_id[c] = char_vocab_size;
            id_to_char[char_vocab_size] = c;
            char_vocab_size++;
        }
    }

    std::vector<int> tokenize(const std::string& word) {
        std::vector<int> char_ids;
        for (char c : word) {
            auto it = char_to_id.find(c);
            char_ids.push_back(it != char_to_id.end() ? it->second : char_unk_id);
        }
        return char_ids;
    }
};

// CRF Layer for structured prediction
struct CRF : torch::nn::Module {
    torch::Tensor transitions; // [num_tags, num_tags]
    int num_tags;

    CRF(int num_tags_) : num_tags(num_tags_) {
        transitions = register_parameter("transitions", torch::randn({num_tags, num_tags}));
    }

    torch::Tensor forward(torch::Tensor emissions, torch::Tensor tags, torch::Tensor mask) {
        // Negative log-likelihood loss
        auto score = compute_score(emissions, tags, mask);
        auto partition = compute_partition(emissions, mask);
        return (partition - score).mean();
    }

    torch::Tensor compute_score(torch::Tensor emissions, torch::Tensor tags, torch::Tensor mask) {
        auto batch_size = emissions.size(0);
        auto seq_len = emissions.size(1);

        torch::Tensor score = torch::zeros({batch_size}, torch::kFloat32).to(emissions.device());
        for (int64_t t = 0; t < seq_len; ++t) {
            auto mask_t = mask.select(1, t); // [batch]
            auto tag_t = tags.select(1, t); // [batch]
            if (t == 0) {
                score += emissions.index({torch::arange(batch_size), t, tag_t}) * mask_t;
            } else {
                auto tag_prev = tags.select(1, t - 1); // [batch]
                score += emissions.index({torch::arange(batch_size), t, tag_t}) * mask_t;
                score += transitions.index({tag_prev, tag_t}) * mask_t;
            }
        }
        return score;
    }

    torch::Tensor compute_partition(torch::Tensor emissions, torch::Tensor mask) {
        auto batch_size = emissions.size(0);
        auto seq_len = emissions.size(1);

        auto alpha = emissions.select(1, 0); // [batch, num_tags]
        for (int64_t t = 1; t < seq_len; ++t) {
            auto mask_t = mask.select(1, t).unsqueeze(1); // [batch, 1]
            auto alpha_t = torch::zeros({batch_size, num_tags}, torch::kFloat32).to(emissions.device());
            for (int64_t j = 0; j < num_tags; ++j) {
                auto broadcast_alpha = alpha + transitions.select(0, j).unsqueeze(0) +
                                      emissions.select(1, t).select(1, j).unsqueeze(1);
                alpha_t.select(1, j) = torch::logsumexp(broadcast_alpha, 1);
            }
            alpha = alpha_t * mask_t + alpha * (1 - mask_t);
        }
        return torch::logsumexp(alpha, 1); // [batch]
    }
};

// BiLSTM-CRF Model
struct BiLSTMCRF : torch::nn::Module {
    torch::nn::Embedding word_embedding{nullptr};
    torch::nn::Embedding char_embedding{nullptr};
    torch::nn::LSTM char_lstm{nullptr};
    torch::nn::LSTM sequence_lstm{nullptr};
    torch::nn::Linear hidden_to_tag{nullptr};
    torch::nn::Dropout dropout{nullptr};
    CRF crf{nullptr};

    BiLSTMCRF(int word_vocab_size, int char_vocab_size, int word_emb_size, int char_emb_size,
              int char_hidden_size, int seq_hidden_size, int num_tags, float dropout_p = 0.1) {
        word_embedding = register_module("word_embedding", torch::nn::Embedding(word_vocab_size, word_emb_size));
        char_embedding = register_module("char_embedding", torch::nn::Embedding(char_vocab_size, char_emb_size));
        char_lstm = register_module("char_lstm", torch::nn::LSTM(torch::nn::LSTMOptions(char_emb_size, char_hidden_size)
                                                                 .num_layers(1)
                                                                 .bidirectional(true)
                                                                 .batch_first(true)));
        sequence_lstm = register_module("sequence_lstm", torch::nn::LSTM(torch::nn::LSTMOptions(word_emb_size + 2 * char_hidden_size, seq_hidden_size)
                                                                         .num_layers(1)
                                                                         .bidirectional(true)
                                                                         .batch_first(true)));
        hidden_to_tag = register_module("hidden_to_tag", torch::nn::Linear(2 * seq_hidden_size, num_tags));
        dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
        crf = register_module("crf", CRF(num_tags));
    }

    torch::Tensor forward(torch::Tensor word_ids, const std::vector<std::vector<std::vector<int>>>& char_ids,
                         torch::Tensor tags, torch::Tensor lengths) {
        auto batch_size = word_ids.size(0);
        auto max_seq_len = word_ids.size(1);

        // Word embeddings
        auto word_emb = word_embedding->forward(word_ids); // [batch, seq_len, word_emb_size]
        word_emb = dropout->forward(word_emb);

        // Character embeddings
        std::vector<torch::Tensor> char_outputs;
        for (size_t b = 0; b < batch_size; ++b) {
            std::vector<torch::Tensor> word_char_embs;
            for (size_t w = 0; w < char_ids[b].size(); ++w) {
                if (char_ids[b][w].empty()) {
                    word_char_embs.push_back(torch::zeros({1, 2 * char_hidden_size}, torch::kFloat32).to(word_ids.device()));
                    continue;
                }
                auto char_tensor = torch::tensor(char_ids[b][w], torch::kInt64).unsqueeze(0).to(word_ids.device());
                auto char_emb = char_embedding->forward(char_tensor); // [1, char_len, char_emb_size]
                auto [char_out, _] = char_lstm->forward(char_emb); // [1, char_len, 2 * char_hidden_size]
                auto char_final = char_out.select(1, -1); // [1, 2 * char_hidden_size]
                word_char_embs.push_back(char_final);
            }
            auto char_seq = torch::cat(word_char_embs, 0); // [seq_len, 2 * char_hidden_size]
            if (char_seq.size(0) < max_seq_len) {
                auto padding = torch::zeros({max_seq_len - char_seq.size(0), 2 * char_hidden_size}, torch::kFloat32).to(word_ids.device());
                char_seq = torch::cat({char_seq, padding}, 0);
            }
            char_outputs.push_back(char_seq.unsqueeze(0)); // [1, seq_len, 2 * char_hidden_size]
        }
        auto char_emb = torch::cat(char_outputs, 0); // [batch, seq_len, 2 * char_hidden_size]
        char_emb = dropout->forward(char_emb);

        // Concatenate word and character embeddings
        auto combined_emb = torch::cat({word_emb, char_emb}, -1); // [batch, seq_len, word_emb_size + 2 * char_hidden_size]

        // Sequence LSTM
        auto packed = torch::nn::utils::rnn::pack_padded_sequence(combined_emb, lengths, true);
        auto [seq_out, _] = sequence_lstm->forward(packed);
        auto [unpacked, _] = torch::nn::utils::rnn::pad_packed_sequence(seq_out, true); // [batch, seq_len, 2 * seq_hidden_size]
        unpacked = dropout->forward(unpacked);

        // Emission scores
        auto emissions = hidden_to_tag->forward(unpacked); // [batch, seq_len, num_tags]

        // CRF loss
        auto mask = (word_ids != word_pad_id).to(torch::kFloat32);
        return crf->forward(emissions, tags, mask);
    }
};

// Sequence Labeling Dataset
struct SequenceLabelingDataset {
    std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> data;
    Vocabulary vocab;
    CharVocabulary char_vocab;
    const int max_len = 50;

    void load_data(const std::string& filename) {
        std::ifstream file(filename);
        std::string line;
        std::vector<std::string> sentences;
        std::vector<std::string> words, tags;
        while (std::getline(file, line)) {
            if (line.empty()) {
                if (!words.empty()) {
                    data.push_back({words, tags});
                    sentences.push_back(std::accumulate(words.begin(), words.end(), std::string(),
                                                        [](const std::string& a, const std::string& b) {
                                                            return a.empty() ? b : a + " " + b;
                                                        }));
                    words.clear();
                    tags.clear();
                }
                continue;
            }
            std::istringstream iss(line);
            std::string word, tag;
            iss >> word >> tag;
            words.push_back(word);
            tags.push_back(tag);
        }
        if (!words.empty()) {
            data.push_back({words, tags});
            sentences.push_back(std::accumulate(words.begin(), words.end(), std::string(),
                                                [](const std::string& a, const std::string& b) {
                                                    return a.empty() ? b : a + " " + b;
                                                }));
        }
        vocab.build(data);
        char_vocab.build(sentences);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<std::vector<std::vector<int>>>> get_batch(
        size_t idx, size_t batch_size) {
        std::vector<torch::Tensor> word_ids_batch, tag_ids_batch, lengths_batch;
        std::vector<std::vector<std::vector<int>>> char_ids_batch;
        size_t max_seq_len = 0;

        for (size_t i = idx; i < std::min(idx + batch_size, data.size()); ++i) {
            max_seq_len = std::max(max_seq_len, std::min(data[i].first.size(), static_cast<size_t>(max_len)));
        }

        for (size_t i = idx; i < std::min(idx + batch_size, data.size()); ++i) {
            auto& [words, tags] = data[i];
            std::vector<int> word_ids, tag_ids;
            std::vector<std::vector<int>> char_ids;
            size_t seq_len = std::min(words.size(), static_cast<size_t>(max_len));

            for (size_t j = 0; j < seq_len; ++j) {
                word_ids.push_back(vocab.get_word_id(words[j]));
                tag_ids.push_back(vocab.get_tag_id(tags[j]));
                char_ids.push_back(char_vocab.tokenize(words[j]));
            }

            while (word_ids.size() < max_seq_len) {
                word_ids.push_back(vocab.word_pad_id);
                tag_ids.push_back(vocab.tag_pad_id);
                char_ids.push_back({});
            }

            word_ids_batch.push_back(torch::tensor(word_ids, torch::kInt64));
            tag_ids_batch.push_back(torch::tensor(tag_ids, torch::kInt64));
            lengths_batch.push_back(torch::tensor(static_cast<int64_t>(seq_len), torch::kInt64));
            char_ids_batch.push_back(char_ids);
        }

        return {
            torch::stack(word_ids_batch),
            torch::stack(tag_ids_batch),
            torch::stack(lengths_batch),
            char_ids_batch
        };
    }
};

int main() {
    torch::manual_seed(0);
    srand(static_cast<unsigned>(time(0)));

    // Load dataset
    SequenceLabelingDataset dataset;
    dataset.load_data("pos_data.txt"); // CoNLL format: word tag per line, empty line separates sentences

    // Initialize model
    BiLSTMCRF model(
        dataset.vocab.word_vocab_size, dataset.char_vocab.char_vocab_size,
        100, 50, 50, 100, dataset.vocab.tag_vocab_size
    );
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    model.to(device);

    // Optimizer
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.001));

    // Training loop
    const int epochs = 20;
    const size_t batch_size = 32;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0;
        size_t num_batches = 0;
        model.train();

        for (size_t i = 0; i < dataset.data.size(); i += batch_size) {
            auto [word_ids, tag_ids, lengths, char_ids] = dataset.get_batch(i, batch_size);
            word_ids = word_ids.to(device);
            tag_ids = tag_ids.to(device);
            lengths = lengths.to(device);

            optimizer.zero_grad();
            auto loss = model.forward(word_ids, char_ids, tag_ids, lengths);
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model.parameters(), 1.0);
            optimizer.step();

            total_loss += loss.item<float>();
            num_batches++;
        }

        std::cout << "Epoch: " << epoch + 1 << ", Loss: " << total_loss / num_batches << std::endl;
    }

    // Save model
    torch::save(model, "stanfordnlp_model.pt");

    return 0;
}




namespace xt::models
{
    StanfordNLP::StanfordNLP(int num_classes, int in_channels)
    {
    }

    StanfordNLP::StanfordNLP(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void StanfordNLP::reset()
    {
    }

    auto StanfordNLP::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];

        return x;
    }
}
