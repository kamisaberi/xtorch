#include <models/natural_language_processing/others/glove.h>


using namespace std;

//
// #include <torch/torch.h>
// #include <vector>
// #include <string>
// #include <fstream>
// #include <sstream>
// #include <random>
// #include <iostream>
// #include <map>
// #include <algorithm>
// #include <cmath>
//
// // Vocabulary class for word-to-ID mapping
// class Vocabulary {
// public:
//     std::map<std::string, int> word_to_id;
//     std::map<int, std::string> id_to_word;
//     int vocab_size = 0;
//     const int unk_id = 0; // <unk>
//
//     Vocabulary() {
//         word_to_id["<unk>"] = unk_id;
//         id_to_word[unk_id] = "<unk>";
//         vocab_size = 1;
//     }
//
//     void build(const std::vector<std::string>& sentences, int min_count = 5) {
//         std::map<std::string, int> word_counts;
//         for (const auto& sentence : sentences) {
//             std::istringstream iss(sentence);
//             std::string word;
//             while (iss >> word) {
//                 word_counts[word]++;
//             }
//         }
//         for (const auto& pair : word_counts) {
//             if (pair.second >= min_count) {
//                 word_to_id[pair.first] = vocab_size;
//                 id_to_word[vocab_size] = pair.first;
//                 vocab_size++;
//             }
//         }
//     }
//
//     int get_id(const std::string& word) const {
//         auto it = word_to_id.find(word);
//         return it != word_to_id.end() ? it->second : unk_id;
//     }
// };
//
// // Co-occurrence matrix dataset
// struct CooccurrenceDataset {
//     std::vector<std::tuple<int, int, float, float>> cooccurrences; // (word_id, context_id, count, weight)
//     Vocabulary vocab;
//     const int window_size = 5;
//     const float x_max = 100.0;
//     const float alpha = 0.75;
//
//     void build(const std::string& filename) {
//         // Load sentences
//         std::ifstream file(filename);
//         std::string line;
//         std::vector<std::string> sentences;
//         while (std::getline(file, line)) {
//             if (!line.empty()) {
//                 sentences.push_back(line);
//             }
//         }
//         vocab.build(sentences);
//
//         // Compute co-occurrence counts
//         std::map<std::pair<int, int>, float> cooc_map;
//         for (const auto& sentence : sentences) {
//             std::istringstream iss(sentence);
//             std::string word;
//             std::vector<int> word_ids;
//             while (iss >> word) {
//                 word_ids.push_back(vocab.get_id(word));
//             }
//             for (size_t i = 0; i < word_ids.size(); ++i) {
//                 int word_id = word_ids[i];
//                 if (word_id == vocab.unk_id) continue;
//                 for (int j = std::max(0, static_cast<int>(i) - window_size);
//                      j <= std::min(static_cast<int>(i) + window_size, static_cast<int>(word_ids.size()) - 1); ++j) {
//                     if (i == j) continue;
//                     int context_id = word_ids[j];
//                     if (context_id == vocab.unk_id) continue;
//                     float distance = std::abs(static_cast<float>(i - j));
//                     float weight = 1.0 / distance; // Decay with distance
//                     cooc_map[{word_id, context_id}] += weight;
//                 }
//             }
//         }
//
//         // Convert to dataset with weights
//         for (const auto& pair : cooc_map) {
//             auto [ids, count] = pair;
//             auto [word_id, context_id] = ids;
//             float weight = std::pow(std::min(count, x_max) / x_max, alpha);
//             cooccurrences.emplace_back(word_id, context_id, count, weight);
//         }
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get_batch(size_t idx, size_t batch_size) {
//         std::vector<int> word_ids, context_ids;
//         std::vector<float> counts, weights;
//         for (size_t i = idx; i < std::min(idx + batch_size, cooccurrences.size()); ++i) {
//             auto [word_id, context_id, count, weight] = cooccurrences[i];
//             word_ids.push_back(word_id);
//             context_ids.push_back(context_id);
//             counts.push_back(std::log(count + 1.0)); // Log of co-occurrence count
//             weights.push_back(weight);
//         }
//         return {
//             torch::tensor(word_ids, torch::kInt64),
//             torch::tensor(context_ids, torch::kInt64),
//             torch::tensor(counts, torch::kFloat32),
//             torch::tensor(weights, torch::kFloat32)
//         };
//     }
// };
//
// // GloVe Model
// struct GloVe : torch::nn::Module {
//     torch::nn::Embedding word_embeddings{nullptr}, context_embeddings{nullptr};
//     torch::nn::Embedding word_biases{nullptr}, context_biases{nullptr};
//
//     GloVe(int vocab_size, int embedding_size) {
//         word_embeddings = register_module("word_embeddings", torch::nn::Embedding(vocab_size, embedding_size));
//         context_embeddings = register_module("context_embeddings", torch::nn::Embedding(vocab_size, embedding_size));
//         word_biases = register_module("word_biases", torch::nn::Embedding(vocab_size, 1));
//         context_biases = register_module("context_biases", torch::nn::Embedding(vocab_size, 1));
//
//         // Initialize embeddings
//         auto emb_init = torch::nn::init::uniform_(word_embeddings->weight, -0.5 / embedding_size, 0.5 / embedding_size);
//         auto ctx_init = torch::nn::init::uniform_(context_embeddings->weight, -0.5 / embedding_size, 0.5 / embedding_size);
//         torch::nn::init::zeros_(word_biases->weight);
//         torch::nn::init::zeros_(context_biases->weight);
//     }
//
//     torch::Tensor forward(torch::Tensor word_ids, torch::Tensor context_ids) {
//         auto word_emb = word_embeddings->forward(word_ids); // [batch, embedding_size]
//         auto context_emb = context_embeddings->forward(context_ids); // [batch, embedding_size]
//         auto word_bias = word_biases->forward(word_ids).squeeze(-1); // [batch]
//         auto context_bias = context_biases->forward(context_ids).squeeze(-1); // [batch]
//
//         // Compute dot product
//         auto dot_product = (word_emb * context_emb).sum(-1); // [batch]
//         return dot_product + word_bias + context_bias; // [batch]
//     }
// };
//
// // Custom GloVe loss function
// struct GloVeLoss : torch::nn::Module {
//     torch::Tensor forward(torch::Tensor predictions, torch::Tensor log_counts, torch::Tensor weights) {
//         auto diff = predictions - log_counts; // [batch]
//         auto weighted_square_error = weights * diff * diff; // [batch]
//         return weighted_square_error.mean();
//     }
// };
//
// int main() {
//     torch::manual_seed(0);
//     srand(static_cast<unsigned>(time(0)));
//
//     // Load dataset
//     CooccurrenceDataset dataset;
//     dataset.build("corpus.txt"); // One sentence per line
//
//     // Initialize model
//     GloVe model(dataset.vocab.vocab_size, 50); // 50-dimensional embeddings
//     torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//     model.to(device);
//
//     // Optimizer and loss
//     torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.05));
//     GloVeLoss loss_fn;
//
//     // Training loop
//     const int epochs = 50;
//     const size_t batch_size = 512;
//     for (int epoch = 0; epoch < epochs; ++epoch) {
//         float total_loss = 0.0;
//         size_t num_batches = 0;
//         model.train();
//
//         for (size_t i = 0; i < dataset.cooccurrences.size(); i += batch_size) {
//             auto [word_ids, context_ids, log_counts, weights] = dataset.get_batch(i, batch_size);
//             word_ids = word_ids.to(device);
//             context_ids = context_ids.to(device);
//             log_counts = log_counts.to(device);
//             weights = weights.to(device);
//
//             optimizer.zero_grad();
//             auto predictions = model.forward(word_ids, context_ids);
//             auto loss = loss_fn.forward(predictions, log_counts, weights);
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
//     // Save model (word embeddings only)
//     torch::save(model.word_embeddings->weight, "glove_embeddings.pt");
//
//     return 0;
// }


namespace xt::models
{
    GloVe::GloVe(int num_classes, int in_channels)
    {
    }

    GloVe::GloVe(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void GloVe::reset()
    {
    }

    auto GloVe::forward(std::initializer_list<std::any> tensors) -> std::any
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
