#include <models/natural_language_processing/others/elmo.h>


using namespace std;


//#include <torch/torch.h>
//#include <vector>
//#include <string>
//#include <fstream>
//#include <sstream>
//#include <random>
//#include <iostream>
//#include <map>
//#include <algorithm>
//
//// Vocabulary class for token-to-ID mapping
//class Vocabulary {
//public:
//    std::map<std::string, int> word_to_id;
//    std::map<int, std::string> id_to_word;
//    int vocab_size = 0;
//    const int unk_id = 0;
//    const int bos_id = 1; // <S>
//    const int eos_id = 2; // </S>
//
//    Vocabulary() {
//        word_to_id["<UNK>"] = unk_id;
//        id_to_word[unk_id] = "<UNK>";
//        word_to_id["<S>"] = bos_id;
//        id_to_word[bos_id] = "<S>";
//        word_to_id["</S>"] = eos_id;
//        id_to_word[eos_id] = "</S>";
//        vocab_size = 3;
//    }
//
//    void build(const std::vector<std::string>& sentences, int min_count = 2) {
//        std::map<std::string, int> word_counts;
//        for (const auto& sentence : sentences) {
//            std::istringstream iss(sentence);
//            std::string word;
//            while (iss >> word) {
//                word_counts[word]++;
//            }
//        }
//        for (const auto& pair : word_counts) {
//            if (pair.second >= min_count) {
//                word_to_id[pair.first] = vocab_size;
//                id_to_word[vocab_size] = pair.first;
//                vocab_size++;
//            }
//        }
//    }
//
//    int get_id(const std::string& word) const {
//        auto it = word_to_id.find(word);
//        return it != word_to_id.end() ? it->second : unk_id;
//    }
//};
//
//// Character CNN for token embeddings
//struct CharCNN : torch::nn::Module {
//    torch::nn::Conv1d conv{nullptr};
//    torch::nn::Linear highway1{nullptr}, highway2{nullptr};
//
//    CharCNN(int char_vocab_size, int char_embed_dim, int output_dim) {
//        conv = register_module("conv", torch::nn::Conv1d(torch::nn::Conv1dOptions(char_embed_dim, 128, 3).padding(1)));
//        highway1 = register_module("highway1", torch::nn::Linear(128, 128));
//        highway2 = register_module("highway2", torch::nn::Linear(128, output_dim));
//    }
//
//    torch::Tensor forward(torch::Tensor x) {
//        x = torch::relu(conv->forward(x));
//        auto max_pool = torch::max_pool1d(x, x.size(2)).squeeze(2);
//        auto h = torch::relu(highway1->forward(max_pool));
//        auto gate = torch::sigmoid(highway1->forward(max_pool));
//        h = gate * h + (1 - gate) * max_pool;
//        h = torch::relu(highway2->forward(h));
//        return h;
//    }
//};
//
//// Bidirectional LSTM for ELMo
//struct BiLSTM : torch::nn::Module {
//    torch::nn::LSTM lstm_forward{nullptr}, lstm_backward{nullptr};
//
//    BiLSTM(int input_size, int hidden_size, int num_layers) {
//        lstm_forward = register_module("lstm_forward", torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size).num_layers(num_layers).batch_first(true)));
//        lstm_backward = register_module("lstm_backward", torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size).num_layers(num_layers).batch_first(true)));
//    }
//
//    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
//        auto x_reverse = torch::flip(x, {1});
//        auto [fwd_output, _] = lstm_forward->forward(x);
//        auto [bwd_output, __] = lstm_backward->forward(x_reverse);
//        bwd_output = torch::flip(bwd_output, {1});
//        return {fwd_output, bwd_output};
//    }
//};
//
//// ELMo Model
//struct ELMo : torch::nn::Module {
//    CharCNN char_cnn{nullptr};
//    BiLSTM bilstm{nullptr};
//    torch::nn::Linear forward_lm{nullptr}, backward_lm{nullptr};
//    torch::Tensor gamma;
//
//    ELMo(int char_vocab_size, int char_embed_dim, int token_embed_dim, int hidden_size, int num_layers, int vocab_size)
//            : char_cnn(CharCNN(char_vocab_size, char_embed_dim, token_embed_dim)),
//              bilstm(BiLSTM(token_embed_dim, hidden_size, num_layers)) {
//        forward_lm = register_module("forward_lm", torch::nn::Linear(hidden_size, vocab_size));
//        backward_lm = register_module("backward_lm", torch::nn::Linear(hidden_size, vocab_size));
//        gamma = register_parameter("gamma", torch::ones(2 * num_layers + 1));
//    }
//
//    std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>> forward(torch::Tensor char_ids) {
//        auto batch_size = char_ids.size(0);
//        auto seq_len = char_ids.size(1);
//        auto max_chars = char_ids.size(2);
//
//        // Character CNN
//        auto char_embed = torch::embedding(torch::randn({261, 16}), char_ids).transpose(1, 3); // [batch, embed, seq, chars]
//        char_embed = char_embed.reshape({batch_size * seq_len, 16, max_chars});
//        auto token_embed = char_cnn.forward(char_embed).reshape({batch_size, seq_len, -1});
//
//        // BiLSTM
//        auto [fwd_output, bwd_output] = bilstm.forward(token_embed);
//
//        // Collect all layers for ELMo embeddings
//        std::vector<torch::Tensor> layers = {token_embed, fwd_output, bwd_output};
//
//        // Language model outputs
//        auto fwd_pred = forward_lm->forward(fwd_output);
//        auto bwd_pred = backward_lm->forward(bwd_output);
//
//        return {fwd_pred, bwd_pred, layers};
//    }
//};
//
//// Dataset class
//struct TextDataset {
//    std::vector<std::vector<int>> sentences;
//    std::vector<std::vector<std::vector<int>>> char_sentences;
//    Vocabulary vocab;
//    const int max_chars = 50;
//
//    void load_data(const std::string& filename) {
//        std::ifstream file(filename);
//        std::string line;
//        std::vector<std::string> raw_sentences;
//        while (std::getline(file, line)) {
//            raw_sentences.push_back(line);
//        }
//        vocab.build(raw_sentences);
//        for (const auto& sentence : raw_sentences) {
//            std::istringstream iss(sentence);
//            std::string word;
//            std::vector<int> token_ids = {vocab.bos_id};
//            std::vector<std::vector<int>> char_ids;
//            char_ids.push_back(std::vector<int>(max_chars, 0)); // <S>
//            while (iss >> word) {
//                token_ids.push_back(vocab.get_id(word));
//                std::vector<int> chars(max_chars, 0);
//                for (size_t i = 0; i < std::min(word.size(), static_cast<size_t>(max_chars)); ++i) {
//                    chars[i] = static_cast<int>(word[i]);
//                }
//                char_ids.push_back(chars);
//            }
//            token_ids.push_back(vocab.eos_id);
//            char_ids.push_back(std::vector<int>(max_chars, 0)); // </S>
//            sentences.push_back(token_ids);
//            char_sentences.push_back(char_ids);
//        }
//    }
//
//    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_batch(size_t idx, size_t batch_size) {
//        std::vector<torch::Tensor> token_inputs, token_targets, char_inputs;
//        size_t max_len = 0;
//        for (size_t i = idx; i < std::min(idx + batch_size, sentences.size()); ++i) {
//            max_len = std::max(max_len, sentences[i].size());
//        }
//        for (size_t i = idx; i < std::min(idx + batch_size, sentences[i].size()); ++i) {
//            std::vector<int> padded_tokens = sentences[i];
//            std::vector<std::vector<int>> padded_chars = char_sentences[i];
//            while (padded_tokens.size() < max_len) {
//                padded_tokens.push_back(vocab.eos_id);
//                padded_chars.push_back(std::vector<int>(max_chars, 0));
//            }
//            token_inputs.push_back(torch::tensor(padded_tokens, torch::kInt64));
//            token_targets.push_back(torch::tensor(std::vector<int>(padded_tokens.begin() + 1, padded_tokens.end()), torch::kInt64));
//            token_targets.back() = torch::cat({token_targets.back(), torch::tensor({vocab.eos_id}, torch::kInt64)});
//            char_inputs.push_back(torch::tensor(padded_chars, torch::kInt64));
//        }
//        return {torch::stack(token_inputs), torch::stack(token_targets), torch::stack(char_inputs)};
//    }
//};
//
//int main() {
//    torch::manual_seed(0);
//    srand(static_cast<unsigned>(time(0)));
//
//    // Load dataset
//    TextDataset dataset;
//    dataset.load_data("corpus.txt"); // Assume a text file with one sentence per line
//
//    // Initialize model
//    ELMo model(261, 16, 128, 256, 2, dataset.vocab.vocab_size);
//    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//    model.to(device);
//
//    // Optimizer
//    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.001));
//
//    // Training loop
//    const int epochs = 10;
//    const size_t batch_size = 32;
//    for (int epoch = 0; epoch < epochs; ++epoch) {
//        float total_loss = 0.0;
//        size_t num_batches = 0;
//        for (size_t i = 0; i < dataset.sentences.size(); i += batch_size) {
//            auto [token_inputs, token_targets, char_inputs] = dataset.get_batch(i, batch_size);
//            char_inputs = char_inputs.to(device);
//            token_targets = token_targets.to(device);
//
//            optimizer.zero_grad();
//            auto [fwd_pred, bwd_pred, _] = model.forward(char_inputs);
//
//            auto fwd_loss = torch::nll_loss(torch::log_softmax(fwd_pred, 2), token_targets);
//            auto bwd_loss = torch::nll_loss(torch::log_softmax(bwd_pred, 2), token_targets);
//            auto loss = fwd_loss + bwd_loss;
//
//            loss.backward();
//            optimizer.step();
//
//            total_loss += loss.item<float>();
//            num_batches++;
//        }
//
//        std::cout << "Epoch: " << epoch + 1 << ", Loss: " << total_loss / num_batches << std::endl;
//    }
//
//    // Save model weights
//    torch::save(model, "elmo_model.pt");
//
//    return 0;
//}

namespace xt::models
{
    ELMo::ELMo(int num_classes, int in_channels)
    {
    }

    ELMo::ELMo(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void ELMo::reset()
    {
    }

    auto ELMo::forward(std::initializer_list<std::any> tensors) -> std::any
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
