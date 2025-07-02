#include "include/models/natural_language_processing/transformers/long_former.h"


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
//
// // Vocabulary class for token-to-ID mapping
// class Vocabulary {
// public:
//     std::map<std::string, int> word_to_id;
//     std::map<int, std::string> id_to_word;
//     int vocab_size = 0;
//     const int unk_id = 0; // [UNK]
//     const int cls_id = 1; // [CLS]
//     const int sep_id = 2; // [SEP]
//     const int mask_id = 3; // [MASK]
//
//     Vocabulary() {
//         word_to_id["[UNK]"] = unk_id;
//         id_to_word[unk_id] = "[UNK]";
//         word_to_id["[CLS]"] = cls_id;
//         id_to_word[cls_id] = "[CLS]";
//         word_to_id["[SEP]"] = sep_id;
//         id_to_word[sep_id] = "[SEP]";
//         word_to_id["[MASK]"] = mask_id;
//         id_to_word[mask_id] = "[MASK]";
//         vocab_size = 4;
//     }
//
//     void build(const std::vector<std::string>& sentences, int min_count = 2) {
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
// // Transformer Embedding Layer
// struct TransformerEmbedding : torch::nn::Module {
//     torch::nn::Embedding word_embeddings{nullptr};
//     torch::nn::Embedding position_embeddings{nullptr};
//     torch::nn::LayerNorm layer_norm{nullptr};
//     torch::nn::Dropout dropout{nullptr};
//
//     TransformerEmbedding(int vocab_size, int hidden_size, int max_position, float dropout_p = 0.1) {
//         word_embeddings = register_module("word_embeddings", torch::nn::Embedding(vocab_size, hidden_size));
//         position_embeddings = register_module("position_embeddings", torch::nn::Embedding(max_position, hidden_size));
//         layer_norm = register_module("layer_norm", torch::nn::LayerNorm(hidden_size));
//         dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
//     }
//
//     torch::Tensor forward(torch::Tensor input_ids) {
//         auto batch_size = input_ids.size(0);
//         auto seq_len = input_ids.size(1);
//         auto embeddings = word_embeddings->forward(input_ids);
//         auto positions = torch::arange(seq_len, torch::kInt64).unsqueeze(0).repeat({batch_size, 1}).to(input_ids.device());
//         embeddings = embeddings + position_embeddings->forward(positions);
//         embeddings = layer_norm->forward(embeddings);
//         embeddings = dropout->forward(embeddings);
//         return embeddings;
//     }
// };
//
// // Longformer Sparse Attention
// struct LongformerAttention : torch::nn::Module {
//     int num_heads;
//     int head_size;
//     int window_size;
//     // Size of sliding window
//     torch::nn::Linear query{nullptr}, key{nullptr}, value{nullptr}, out{nullptr};
//     torch::nn::Dropout dropout{nullptr};
//
//     LongformerAttention(int hidden_size, int num_heads, int window_size, float dropout_p = 0.1)
//         : num_heads(num_heads), head_size(hidden_size / num_heads), window_size(window_size) {
//         query = register_module("query", torch::nn::Linear(hidden_size, hidden_size));
//         key = register_module("key", torch::nn::Linear(hidden_size, hidden_size));
//         value = register_module("value", torch::nn::Linear(hidden_size, hidden_size));
//         out = register_module("out", torch::nn::Linear(hidden_size, hidden_size));
//         dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
//     }
//
//     torch::Tensor forward(torch::Tensor x, torch::Tensor attention_mask = {}, torch::Tensor global_attention_mask = nullptr) {
//         auto batch_size = x.size(0);
//         auto seq_len = x.size(1);
//
//         auto q = query->forward(x).view({batch_size, seq_len, num_heads, head_size}).transpose(1, 2);
//         auto k = key->forward(x).view({batch_size, seq_len, num_heads, head_size}).transpose(1, 2);
//         auto v = value->forward(x).view({batch_size, seq_len, num_heads, head_size}).transpose(1, 2);
//
//         // Create sparse attention mask
//         auto sparse_mask = create_sparse_mask(batch_size, seq_len);
//         if (attention_mask.defined()) {
//             sparse_mask = sparse_mask.to(attention_mask.device()) * attention_mask;
//         }
//
//         // Apply global attention mask (e.g., [CLS] attends to all attend to [CLS])
//         if (global_attention_mask.defined()) {
//             sparse_mask = sparse_mask.where(global_attention_mask == 0, torch::ones_like(sparse_mask));
//         }
//
//         auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<float>(head_size));
//         scores = scores + (sparse_mask * -1e9);
//         scores = torch::softmax(scores, -1);
//         scores = dropout->forward(scores);
//         auto context = torch::matmul(scores, v).transpose(1, 2).contiguous().view({batch_size, seq_len, -1});
//         return out->forward(context);
//     }
//
// private:
//     torch::Tensor create_sparse_mask(int batch_size, int seq_len) {
//         auto mask = torch::zeros({batch_size, num_heads, seq_len, seq_len}, torch::kFloat32);
//
//         // Sliding window attention
//         for (int i = 0; i < seq_len; ++i) {
//             int start = std::max(0, i - window_size / 2);
//             int end = std::min(seq_len, i + window_size / 2 + 1);
//             mask.slice(2, i, i + 1).slice(3, start, end).fill_(1.0);
//         }
//
//         // Global attention for [CLS] (index 0)
//         mask.slice(2, 0, 1).fill_(1.0); // [CLS] attends to all
//         mask.slice(3, 0, 1).fill_(1.0); // All attend to [CLS]
//
//         return mask;
//     }
// };
//
// // Feed-Forward Network
// struct FeedForward : torch::nn::Module {
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr};
//     torch::nn::Dropout dropout{nullptr};
//
//     FeedForward(int hidden_size, int intermediate_size, float dropout_p = 0.1) {
//         fc1 = register_module("fc1", torch::nn::Linear(hidden_size, intermediate_size));
//         fc2 = register_module("fc2", torch::nn::Linear(intermediate_size, hidden_size));
//         dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::gelu(fc1->forward(x));
//         x = dropout->forward(x);
//         x = fc2->forward(x);
//         return x;
//     }
// };
//
// // Transformer Layer
// struct TransformerLayer : torch::nn::Module {
//     LongformerAttention attention{nullptr};
//     torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
//     FeedForward ffn{nullptr};
//     torch::nn::Dropout dropout{nullptr};
//
//     TransformerLayer(int hidden_size, int num_heads, int intermediate_size, int window_size, float dropout_p = 0.1)
//         : attention(LongformerAttention(hidden_size, num_heads, window_size, dropout_p)),
//           ffn(FeedForward(hidden_size, intermediate_size, dropout_p)) {
//         norm1 = register_module("norm1", torch::nn::LayerNorm(hidden_size));
//         norm2 = register_module("norm2", torch::nn::LayerNorm(hidden_size));
//         dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
//     }
//
//     torch::Tensor forward(torch::Tensor x, torch::Tensor attention_mask = {}, torch::Tensor global_attention_mask = nullptr) {
//         auto attn_output = attention->forward(x, attention_mask, global_attention_mask);
//         x = norm1->forward(x + dropout->forward(attn_output));
//         auto ffn_output = ffn->forward(x);
//         x = norm2->forward(x + dropout->forward(ffn_output));
//         return x;
//     }
// };
//
// // Longformer Model
// struct Longformer : torch::nn::Module {
//     TransformerEmbedding embedding{nullptr};
//     torch::nn::ModuleList layers{nullptr};
//     torch::nn::Linear mlm_head{nullptr};
//     torch::nn::LayerNorm mlm_norm{nullptr};
//
//     Longformer(int vocab_size, int hidden_size, int num_heads, int intermediate_size, int num_layers, int max_position, int window_size) {
//         embedding = register_module("embedding", TransformerEmbedding(vocab_size, hidden_size, max_position));
//         layers = register_module("layers", torch::nn::ModuleList());
//         for (int i = 0; i < num_layers; ++i) {
//             layers->push_back(TransformerLayer(hidden_size, num_heads, intermediate_size, window_size));
//         }
//         mlm_head = register_module("mlm_head", torch::nn::Linear(hidden_size, vocab_size));
//         mlm_norm = register_module("mlm_norm", torch::nn::LayerNorm(hidden_size));
//     }
//
//     torch::Tensor forward(torch::Tensor input_ids, torch::Tensor attention_mask = {}, torch::Tensor global_attention_mask = nullptr) {
//         auto x = embedding->forward(input_ids);
//         for (auto& layer : *layers) {
//             x = layer->as<TransformerLayer>()->forward(x, attention_mask, global_attention_mask);
//         }
//         x = mlm_norm->forward(x);
//         return mlm_head->forward(x);
//     }
// };
//
// // Dataset for MLM
// struct MLMDataset {
//     std::vector<std::vector<int>> sentences;
//     Vocabulary vocab;
//     const int max_len = 512; // Longer sequences for Longformer
//     const float mask_prob = 0.15;
//
//     void load_data(const std::string& filename) {
//         std::ifstream file(filename);
//         std::string line;
//         std::vector<std::string> raw_sentences;
//         while (std::getline(file, line)) {
//             raw_sentences.push_back(line);
//         }
//         vocab.build(raw_sentences);
//         for (const auto& sentence : raw_sentences) {
//             std::istringstream iss(sentence);
//             std::string word;
//             std::vector<int> tokens = {vocab.cls_id};
//             while (iss >> word && tokens.size() < max_len - 1) {
//                 tokens.push_back(vocab.get_id(word));
//             }
//             tokens.push_back(vocab.sep_id);
//             sentences.push_back(tokens);
//         }
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get_batch(size_t idx, size_t batch_size) {
//         std::vector<torch::Tensor> inputs, targets, attention_masks, global_attention_masks;
//         std::random_device rd;
//         std::mt19937 gen(rd());
//         std::uniform_real_distribution<float> dist(0.0, 1.0);
//         std::uniform_int_distribution<int> word_dist(4, vocab.vocab_size - 1);
//
//         size_t max_batch_len = 0;
//         for (size_t i = idx; i < std::min(idx + batch_size, sentences.size()); ++i) {
//             max_batch_len = std::max(max_batch_len, sentences[i].size());
//         }
//
//         for (size_t i = idx; i < std::min(idx + batch_size, sentences.size()); ++i) {
//             auto tokens = sentences[i];
//             std::vector<int> input_tokens = tokens;
//             std::vector<int> target_tokens(max_batch_len, -100); // -100 for ignored indices in cross_entropy
//             std::vector<int> attention_mask(max_batch_len, 1);
//             std::vector<int> global_attention_mask(max_batch_len, 0);
//
//             // Mask tokens
//             for (size_t j = 1; j < tokens.size() - 1; ++j) { // Skip [CLS] and [SEP]
//                 if (dist(gen) < mask_prob) {
//                     float r = dist(gen);
//                     if (r < 0.8) {
//                         input_tokens[j] = vocab.mask_id; // 80% MASK
//                     } else if (r < 0.9) {
//                         input_tokens[j] = word_dist(gen); // 10% random word
//                     } // 10% keep original
//                     target_tokens[j] = tokens[j];
//                 }
//             }
//
//             // Set global attention for [CLS]
//             global_attention_mask[0] = 1;
//
//             while (input_tokens.size() < max_batch_len) {
//                 input_tokens.push_back(0);
//                 attention_mask.push_back(0);
//                 global_attention_mask.push_back(0);
//             }
//
//             inputs.push_back(torch::tensor(input_tokens, torch::kInt64));
//             targets.push_back(torch::tensor(target_tokens, torch::kInt64));
//             attention_masks.push_back(torch::tensor(attention_mask, torch::kFloat32));
//             global_attention_masks.push_back(torch::tensor(global_attention_mask, torch::kFloat32));
//         }
//
//         return {
//             torch::stack(inputs),
//             torch::stack(targets),
//             torch::stack(attention_masks),
//             torch::stack(global_attention_masks)
//         };
//     }
// };
//
// int main() {
//     torch::manual_seed(0);
//     srand(static_cast<unsigned>(time(0)));
//
//     // Load dataset
//     MLMDataset dataset;
//     dataset.load_data("corpus.txt"); // One sentence per line
//
//     // Initialize model
//     Longformer model(dataset.vocab.vocab_size, 256, 4, 1024, 6, 512, 64);
//     torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//     model.to(device);
//
//     // Optimizer
//     torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.0001).weight_decay(0.01));
//
//     // Training loop
//     const int epochs = 10;
//     const size_t batch_size = 16;
//     for (int epoch = 0; epoch < epochs; ++epoch) {
//         float total_loss = 0.0;
//         size_t num_batches = 0;
//         for (size_t i = 0; i < dataset.sentences.size(); i += batch_size) {
//             auto [input_ids, labels, attention_mask, global_attention_mask] = dataset.get_batch(i, batch_size);
//             input_ids = input_ids.to(device);
//             labels = labels.to(device);
//             attention_mask = attention_mask.to(device).unsqueeze(1).unsqueeze(2);
//             global_attention_mask = global_attention_mask.to(device).unsqueeze(1).unsqueeze(2);
//
//             optimizer.zero_grad();
//             auto logits = model.forward(input_ids, attention_mask, global_attention_mask);
//             auto loss = torch::cross_entropy(logits.view({-1, dataset.vocab.vocab_size}), labels.view(-1));
//             loss.backward();
//             optimizer.step();
//
//             total_loss += loss.item<float>();
//             num_batches++;
//         }
//         std::cout << "Epoch: " << epoch + 1 << ", Loss: " << total_loss / num_batches << std::endl;
//     }
//
//     // Save model
//     torch::save(model, "longformer_model.pt");
//
//     return 0;
// }



namespace xt::models
{
    Longformer::Longformer(int num_classes, int in_channels)
    {
    }

    Longformer::Longformer(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void Longformer::reset()
    {
    }

    auto Longformer::forward(std::initializer_list<std::any> tensors) -> std::any
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
