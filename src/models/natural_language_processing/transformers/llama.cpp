#include "include/models/natural_language_processing/transformers/llama.h"


using namespace std;


// #include <torch/torch.h>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
// #include <fstream>
// #include <map>
// #include <random>
// #include <sstream>
//
// // Vocabulary Class
// class Vocabulary {
// public:
//     Vocabulary(const std::string& vocab_file) {
//         std::ifstream file(vocab_file);
//         std::string token;
//         int idx = 0;
//         while (std::getline(file, token)) {
//             token_to_idx_[token] = idx;
//             idx_to_token_[idx] = token;
//             idx++;
//         }
//         vocab_size_ = idx;
//     }
//
//     int token_to_idx(const std::string& token) const {
//         auto it = token_to_idx_.find(token);
//         return it != token_to_idx_.end() ? it->second : token_to_idx_.at("<unk>");
//     }
//
//     std::string idx_to_token(int idx) const {
//         auto it = idx_to_token_.find(idx);
//         return it != idx_to_token_.end() ? it->second : "<unk>";
//     }
//
//     int vocab_size() const { return vocab_size_; }
//
// private:
//     std::map<std::string, int> token_to_idx_;
//     std::map<int, std::string> idx_to_token_;
//     int vocab_size_;
// };
//
// // Custom Prompt-Response Dataset
// class PromptResponseDataset : public torch::data::Dataset<PromptResponseDataset> {
// public:
//     PromptResponseDataset(const std::string& prompt_dir, const std::string& response_dir, const Vocabulary& vocab, int max_seq_length)
//         : vocab_(vocab), max_seq_length_(max_seq_length) {
//         for (const auto& entry : std::filesystem::directory_iterator(prompt_dir)) {
//             if (entry.path().extension() == ".txt") {
//                 prompt_paths_.push_back(entry.path().string());
//                 std::string response_path = response_dir + "/" + entry.path().filename().string();
//                 response_paths_.push_back(response_path);
//             }
//         }
//     }
//
//     torch::data::Example<> get(size_t index) override {
//         std::ifstream prompt_file(prompt_paths_[index]);
//         if (!prompt_file.is_open()) {
//             throw std::runtime_error("Failed to open prompt file: " + prompt_paths_[index]);
//         }
//         std::string prompt;
//         std::getline(prompt_file, prompt);
//
//         std::ifstream response_file(response_paths_[index]);
//         if (!response_file.is_open()) {
//             throw std::runtime_error("Failed to open response file: " + response_paths_[index]);
//         }
//         std::string response;
//         std::string line;
//         while (std::getline(response_file, line)) {
//             response += line + " ";
//         }
//
//         std::vector<std::string> tokens;
//         tokens.push_back("<sos>");
//         std::istringstream prompt_iss(prompt);
//         std::string token;
//         while (prompt_iss >> token && tokens.size() < max_seq_length_ / 2) {
//             tokens.push_back(token);
//         }
//         tokens.push_back("<sep>");
//         std::istringstream response_iss(response);
//         while (response_iss >> token && tokens.size() < max_seq_length_ - 1) {
//             tokens.push_back(token);
//         }
//         tokens.push_back("<eos>");
//         while (tokens.size() < max_seq_length_) {
//             tokens.push_back("<pad>");
//         }
//
//         std::vector<int64_t> token_indices;
//         for (const auto& t : tokens) {
//             token_indices.push_back(vocab_.token_to_idx(t));
//         }
//         torch::Tensor input_tensor = torch::tensor(token_indices, torch::kInt64);
//
//         std::vector<int64_t> target_indices(token_indices.begin() + 1, token_indices.end());
//         target_indices.pop_back();
//         target_indices.push_back(vocab_.token_to_idx("<pad>"));
//         torch::Tensor target_tensor = torch::tensor(target_indices, torch::kInt64);
//
//         return {input_tensor, target_tensor};
//     }
//
//     torch::optional<size_t> size() const override {
//         return prompt_paths_.size();
//     }
//
// private:
//     std::vector<std::string> prompt_paths_, response_paths_;
//     const Vocabulary& vocab_;
//     int max_seq_length_;
// };
//
// // Grouped-Query Attention (GQA)
// struct GroupedQueryAttentionImpl : torch::nn::Module {
//     GroupedQueryAttentionImpl(int d_model, int nhead, int n_kv_groups, float dropout)
//         : nhead_(nhead), n_kv_groups_(n_kv_groups), d_head_(d_model / nhead) {
//         qkv_proj = register_module("qkv_proj", torch::nn::Linear(d_model, d_model * 3));
//         out_proj = register_module("out_proj", torch::nn::Linear(d_model, d_model));
//         dropout = register_module("dropout", torch::nn::Dropout(dropout));
//     }
//
//     torch::Tensor forward(torch::Tensor x, torch::Tensor mask = {}) {
//         auto batch_size = x.size(0);
//         auto seq_len = x.size(1);
//         auto qkv = qkv_proj->forward(x).view({batch_size, seq_len, 3, nhead_, d_head_}).permute({2, 0, 3, 1, 4});
//         auto q = qkv[0]; // [batch, nhead, seq_len, d_head]
//         auto k = qkv[1]; // [batch, nhead, seq_len, d_head]
//         auto v = qkv[2]; // [batch, nhead, seq_len, d_head]
//
//         // Group keys and values
//         auto k_grouped = k.view({batch_size, n_kv_groups_, nhead_ / n_kv_groups_, seq_len, d_head_});
//         auto v_grouped = v.view({batch_size, n_kv_groups_, nhead_ / n_kv_groups_, seq_len, d_head_});
//         k_grouped = k_grouped.mean(2).unsqueeze(2).expand({batch_size, n_kv_groups_, nhead_ / n_kv_groups_, seq_len, d_head_}).contiguous().view({batch_size, nhead_, seq_len, d_head_});
//         v_grouped = v_grouped.mean(2).unsqueeze(2).expand({batch_size, n_kv_groups_, nhead_ / n_kv_groups_, seq_len, d_head_}).contiguous().view({batch_size, nhead_, seq_len, d_head_});
//
//         auto scores = torch::matmul(q, k_grouped.transpose(2, 3)) / std::sqrt(static_cast<float>(d_head_));
//         if (!mask.is_empty()) {
//             scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(1), -1e9);
//         }
//         auto attn_weights = torch::softmax(scores, -1);
//         attn_weights = dropout->forward(attn_weights);
//         auto attn_output = torch::matmul(attn_weights, v_grouped);
//         attn_output = attn_output.permute({0, 2, 1, 3}).contiguous().view({batch_size, seq_len, nhead_ * d_head_});
//         return out_proj->forward(attn_output);
//     }
//
//     int nhead_, n_kv_groups_, d_head_;
//     torch::nn::Linear qkv_proj{nullptr}, out_proj{nullptr};
//     torch::nn::Dropout dropout{nullptr};
// };
// TORCH_MODULE(GroupedQueryAttention);
//
// // Transformer Decoder Layer
// struct TransformerDecoderLayerImpl : torch::nn::Module {
//     TransformerDecoderLayerImpl(int d_model, int nhead, int n_kv_groups, int dim_feedforward, float dropout) {
//         self_attn = register_module("self_attn", GroupedQueryAttention(d_model, nhead, n_kv_groups, dropout));
//         linear1 = register_module("linear1", torch::nn::Linear(d_model, dim_feedforward));
//         dropout1 = register_module("dropout1", torch::nn::Dropout(dropout));
//         linear2 = register_module("linear2", torch::nn::Linear(dim_feedforward, d_model));
//         norm1 = register_module("norm1", torch::nn::LayerNorm(d_model));
//         norm2 = register_module("norm2", torch::nn::LayerNorm(d_model));
//         dropout2 = register_module("dropout2", torch::nn::Dropout(dropout));
//     }
//
//     torch::Tensor forward(torch::Tensor tgt, torch::Tensor tgt_mask = {}) {
//         tgt = norm1->forward(tgt); // Pre-LN
//         auto attn_out = self_attn->forward(tgt, tgt_mask);
//         tgt = tgt + dropout1->forward(attn_out);
//         tgt = norm2->forward(tgt); // Pre-LN
//         auto ff_out = linear2->forward(torch::swish(linear1->forward(tgt))); // Swish activation
//         tgt = tgt + dropout2->forward(ff_out);
//         return tgt;
//     }
//
//     GroupedQueryAttention self_attn{nullptr};
//     torch::nn::Linear linear1{nullptr}, linear2{nullptr};
//     torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr};
//     torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
// };
// TORCH_MODULE(TransformerDecoderLayer);
//
// // Transformer Decoder
// struct TransformerDecoderImpl : torch::nn::Module {
//     TransformerDecoderImpl(int d_model, int nhead, int n_kv_groups, int num_layers, int dim_feedforward, float dropout) {
//         for (int i = 0; i < num_layers; ++i) {
//             layers->push_back(TransformerDecoderLayer(d_model, nhead, n_kv_groups, dim_feedforward, dropout));
//             register_module("layer_" + std::to_string(i), layers->back());
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor tgt, torch::Tensor tgt_mask = {}) {
//         torch::Tensor output = tgt;
//         for (auto& layer : *layers) {
//             output = layer->forward(output, tgt_mask);
//         }
//         return output;
//     }
//
//     torch::nn::ModuleList layers{torch::nn::ModuleList()};
// };
// TORCH_MODULE(TransformerDecoder);
//
// // Simplified LLaMA-2-like Model
// struct LLaMA2Impl : torch::nn::Module {
//     LLaMA2Impl(int vocab_size, int d_model = 1024, int nhead = 16, int n_kv_groups = 4, int num_layers = 12, int dim_feedforward = 4096, float dropout = 0.1) {
//         token_embedding = register_module("token_embedding", torch::nn::Embedding(vocab_size, d_model));
//         positional_encoding = register_parameter("positional_encoding",
//             torch::randn({1, 4096, d_model})); // LLaMA-2 context length
//         decoder = register_module("decoder", TransformerDecoder(d_model, nhead, n_kv_groups, num_layers, dim_feedforward, dropout));
//         output_layer = register_module("output_layer", torch::nn::Linear(d_model, vocab_size));
//     }
//
//     torch::Tensor forward(torch::Tensor input) {
//         auto features = token_embedding->forward(input); // [batch, seq_len, d_model]
//         features = features * std::sqrt(static_cast<float>(input.size(2))); // Scale embeddings
//         features = features + positional_encoding.slice(1, 0, input.size(1));
//         auto causal_mask = torch::triu(torch::ones({input.size(1), input.size(1)}), 1).to(torch::kBool);
//         auto decoded = decoder->forward(features, causal_mask); // [batch, seq_len, d_model]
//         return output_layer->forward(decoded); // [batch, seq_len, vocab_size]
//     }
//
//     torch::nn::Embedding token_embedding{nullptr};
//     torch::Tensor positional_encoding;
//     TransformerDecoder decoder{nullptr};
//     torch::nn::Linear output_layer{nullptr};
// };
// TORCH_MODULE(LLaMA2);
//
// int main() {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Hyperparameters
//         const int batch_size = 8;
//         const int num_epochs = 10;
//         const float learning_rate = 0.0001;
//         const int d_model = 1024;
//         const int nhead = 16;
//         const int n_kv_groups = 4; // GQA: 4 groups for 16 heads
//         const int num_layers = 12;
//         const int dim_feedforward = 4096;
//         const float dropout = 0.1;
//         const int max_seq_length = 512;
//
//         // Load vocabulary
//         Vocabulary vocab("./data/vocab.txt");
//
//         // Initialize model
//         LLaMA2 model(vocab.vocab_size(), d_model, nhead, n_kv_groups, num_layers, dim_feedforward, dropout);
//         model->to(device);
//
//         // Optimizer
//         torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(learning_rate).weight_decay(0.1));
//
//         // Dataset and DataLoader
//         auto dataset = PromptResponseDataset("./data/prompts", "./data/responses", vocab, max_seq_length)
//             .map(torch::data::transforms::Stack<>());
//         auto data_loader = torch::data::make_data_loader(
//             std::move(dataset),
//             torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));
//
//         // Training loop
//         model->train();
//         for (int epoch = 0; epoch < num_epochs; ++epoch) {
//             float total_loss = 0.0;
//             int batch_count = 0;
//             for (auto& batch : *data_loader) {
//                 auto inputs = batch.data.to(device);
//                 auto targets = batch.target.to(device);
//
//                 optimizer.zero_grad();
//                 auto outputs = model->forward(inputs);
//                 auto loss = torch::nn::functional::cross_entropy(
//                     outputs.view({-1, vocab.vocab_size()}),
//                     targets.view({-1})
//                 );
//                 loss.backward();
//                 torch::nn::utils::clip_grad_norm_(model->parameters(), 0.5); // Gradient clipping
//                 optimizer.step();
//
//                 total_loss += loss.item<float>();
//                 batch_count++;
//             }
//
//             float avg_loss = total_loss / batch_count;
//             std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "] "
//                       << "Loss: " << avg_loss << std::endl;
//
//             // Save checkpoint every 5 epochs
//             if ((epoch + 1) % 5 == 0) {
//                 torch::save(model, "llama2_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: llama2_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "llama2_final.pt");
//         std::cout << "Saved final model: llama2_final.pt" << std::endl;
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
//
//     return 0;
// }


namespace xt::models
{
    LLAMA::LLAMA(int num_classes, int in_channels)
    {
    }

    LLAMA::LLAMA(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void LLAMA::reset()
    {
    }

    auto LLAMA::forward(std::initializer_list<std::any> tensors) -> std::any
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
