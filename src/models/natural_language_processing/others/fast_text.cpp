#include "include/models/natural_language_processing/others/fast_text.h"


using namespace std;

// #include <torch/torch.h>
// #include <vector>
// #include <string>
// #include <fstream>
// #include <sstream>
// #include <random>
// #include <iostream>
// #include <map>
// #include <algorithm>
// #include <set>
//
// // Vocabulary class for word and subword (n-gram) to ID mapping
// class Vocabulary {
// public:
//     std::map<std::string, int> token_to_id;
//     std::map<int, std::string> id_to_token;
//     int vocab_size = 0;
//     const int unk_id = 0; // <unk>
//     const int pad_id = 1; // <pad>
//     int min_ngram = 3;
//     int max_ngram = 6;
//
//     Vocabulary() {
//         token_to_id["<unk>"] = unk_id;
//         id_to_token[unk_id] = "<unk>";
//         token_to_id["<pad>"] = pad_id;
//         id_to_token[pad_id] = "<pad>";
//         vocab_size = 2;
//     }
//
//     // Generate character n-grams for a word
//     std::vector<std::string> get_ngrams(const std::string& word) {
//         std::vector<std::string> ngrams;
//         std::string padded_word = "<" + word + ">";
//         for (int len = min_ngram; len <= max_ngram; ++len) {
//             for (size_t i = 0; i + len <= padded_word.size(); ++i) {
//                 ngrams.push_back(padded_word.substr(i, len));
//             }
//         }
//         ngrams.push_back(word); // Include the word itself
//         return ngrams;
//     }
//
//     void build(const std::vector<std::string>& sentences, int min_count = 1) {
//         std::map<std::string, int> token_counts;
//         for (const auto& sentence : sentences) {
//             std::istringstream iss(sentence);
//             std::string word;
//             while (iss >> word) {
//                 auto ngrams = get_ngrams(word);
//                 for (const auto& ngram : ngrams) {
//                     token_counts[ngram]++;
//                 }
//             }
//         }
//         for (const auto& pair : token_counts) {
//             if (pair.second >= min_count) {
//                 token_to_id[pair.first] = vocab_size;
//                 id_to_token[vocab_size] = pair.first;
//                 vocab_size++;
//             }
//         }
//     }
//
//     int get_id(const std::string& token) const {
//         auto it = token_to_id.find(token);
//         return it != token_to_id.end() ? it->second : unk_id;
//     }
//
//     // Tokenize sentence into n-gram IDs
//     std::vector<int> tokenize(const std::string& sentence) {
//         std::vector<int> token_ids;
//         std::istringstream iss(sentence);
//         std::string word;
//         while (iss >> word) {
//             auto ngrams = get_ngrams(word);
//             for (const auto& ngram : ngrams) {
//                 token_ids.push_back(get_id(ngram));
//             }
//         }
//         return token_ids;
//     }
// };
//
// // FastText Model
// struct FastText : torch::nn::Module {
//     torch::nn::Embedding embedding{nullptr};
//     torch::nn::Linear classifier{nullptr};
//     torch::nn::Dropout dropout{nullptr};
//
//     FastText(int vocab_size, int embedding_size, int num_classes, float dropout_p = 0.1) {
//         embedding = register_module("embedding", torch::nn::Embedding(vocab_size, embedding_size));
//         classifier = register_module("classifier", torch::nn::Linear(embedding_size, num_classes));
//         dropout = register_module("dropout", torch::nn::Dropout_p));
//     }
//
//     torch::Tensor forward(const std::vector<std::vector<int>>& batch_indices, torch::Tensor lengths) {
//         std::vector<torch::Tensor> embedded;
//         for (size_t i = 0; i < batch_indices.size(); ++i) {
//             if (batch_indices[i].empty()) {
//                 embedded.push_back(torch::zeros({1, embedding_size}, torch::kFloat32).to(device.device()));
//                 continue;
//             }
//             auto indices = torch::tensor(batch_indices[i], torch::kInt64).to(device.device());
//             auto emb = embedding->forward(indices); // [seq_len, embedding_size]
//             auto mean_emb = emb.mean(dim=0).unsqueeze(0); // Average over sequence [1, embedding_size]
//             embedded.push_back(mean_emb);
//         }
//         auto x = torch::cat(embedded, 0); // [batch_size, embedding_size]
//         x = dropout->forward(x);
//         x = classifier->forward(x); // [batch_size, num_classes]
//         return x;
//     }
// };
//
// // Text Classification Dataset
// struct TextClassification {
//     std::vector<std::pair<std::string, int>> labeled_sentences;
//     Vocabulary vocab;
//     const int max_len = 512;
//     void load_data(const std::string& filename) {
//         std::ifstream file(filename);
//         std::string line;
//         std::vector<std::string> sentences;
//         while (std::getline(file, line)) {
//             std::istringstream iss(line);
//             int label;
//             iss >> label;
//             std::string sentence;
//             std::getline(iss, sentence);
//             sentence.erase(0, sentence.find_first_not_of(" \t"));
//             if (!sentence.empty()) {
//                 labeled_sentences.push_back({sentence, label});
//                 sentences.push_back(sentence);
//             }
//         }
//         vocab.build(sentences);
//     }
//
//     std::tuple<std::vector<std::vector<int>>, torch::Tensor> get_batch(size_t idx, size_t batch_size) {
//         std::vector<std::vector<int>> token_ids_batch;
//         std::vector<int> labels;
//         for (size_t i = idx; i < std::min(idx + batch_size, labeled_sentences.size()); ++i) {
//             auto& [sentence, label] = labeled_sentences[i];
//             auto token_ids = vocab.tokenize(sentence);
//             if (token_ids.size() > max_len) {
//                 token_ids.resize(max_len);
//             }
//             token_ids_batch.push_back(token_ids);
//             labels.push_back(label);
//         }
//         auto labels_tensor = torch::tensor(labels, torch::kInt64);
//         return {token_ids_batch, labels_tensor};
//     }
// };
//
// int main() {
//     torch::manual_seed(0);
//     srand(static_cast<unsigned>(time(0)));
//
//     // Load dataset
//     TextClassificationDataset dataset;
//     dataset.load_data("data.txt"); // Format: label (0/1) followed by sentence
//
//     // Initialize model
//     FastText model(dataset.vocab.vocab_size, 100, 2); // 2 classes: positive, negative
//     torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//     model.to(device);
//
//     // Optimizer and loss function
//     torch::optim::Adam optimizer(model.parameters(), torch::AdamOptions(0.01));
//     auto criterion = torch::nn::CrossEntropyLoss();
//
//     // Training loop
//     const int epochs = 10;
//     const size_t batch_size = 32;
//     for (int epoch = 0; epoch < epochs; ++epoch) {
//         float total_loss = 0.0;
//         size_t num_batches = 0;
//         model.train();
//
//         for (size_t i = 0; i < dataset.labeled_sentences.size(); i += batch_size) {
//             auto [token_ids_batch, labels] = dataset.get_batch(i, batch_size);
//             labels = labels.to(device);
//
//             optimizer.zero_grad();
//             auto logits = model.forward(token_ids_batch, torch::none);
//             auto loss = criterion->forward(logits, labels);
//             loss.backward();
//             optimizer.step();
//
//             total_loss += loss.item<float>();
//             num_batches++;
//         }
//
//         std::cout << "Epoch: " << epoch + 1 << ", Loss: " << total_loss / num_batches << std::endl;
//     }
//
//     // Save model
//     torch::save(model, "fasttext_model.pt");
//
//     return 0;
// }

namespace xt::models
{
    FastText::FastText(int num_classes, int in_channels)
    {
    }

    FastText::FastText(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void FastText::reset()
    {
    }

    auto FastText::forward(std::initializer_list<std::any> tensors) -> std::any
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
