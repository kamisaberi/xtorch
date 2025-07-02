#include "include/models/natural_language_processing/transformers/grok.h"


using namespace std;

//Grok V1

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
// // Expert Feedforward Network
// struct ExpertFFNImpl : torch::nn::Module {
//     ExpertFFNImpl(int d_model, int dim_feedforward, float dropout) {
//         linear1 = register_module("linear1", torch::nn::Linear(d_model, dim_feedforward));
//         dropout1 = register_module("dropout1", torch::nn::Dropout(dropout));
//         linear2 = register_module("linear2", torch::nn::Linear(dim_feedforward, d_model));
//         dropout2 = register_module("dropout2", torch::nn::Dropout(dropout));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::gelu(linear1->forward(x));
//         x = dropout1->forward(x);
//         x = linear2->forward(x);
//         x = dropout2->forward(x);
//         return x;
//     }
//
//     torch::nn::Linear linear1{nullptr}, linear2{nullptr};
//     torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr};
// };
// TORCH_MODULE(ExpertFFN);
//
// // MoE Layer
// struct MoELayerImpl : torch::nn::Module {
//     MoELayerImpl(int d_model, int num_experts, int dim_feedforward, float dropout) : num_experts_(num_experts) {
//         gate = register_module("gate", torch::nn::Linear(d_model, num_experts));
//         for (int i = 0; i < num_experts; ++i) {
//             experts->push_back(ExpertFFN(d_model, dim_feedforward, dropout));
//             register_module("expert_" + std::to_string(i), experts->back());
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto gate_logits = gate->forward(x); // [batch, seq_len, num_experts]
//         auto gate_probs = torch::softmax(gate_logits, -1);
//         auto [top_probs, top_indices] = gate_probs.topk(2, -1, true, true); // Top-2 experts
//         top_probs = torch::softmax(top_probs, -1); // Normalize top-2
//
//         torch::Tensor output = torch::zeros_like(x);
//         for (int i = 0; i < num_experts_; ++i) {
//             auto expert_mask = (top_indices == i).any(-1, true); // [batch, seq_len, 1]
//             if (expert_mask.sum().item<int64_t>() > 0) {
//                 auto expert_input = x.masked_select(expert_mask).view({-1, x.size(-1)});
//                 auto expert_output = experts->at(i)->forward(expert_input);
//                 auto expanded_output = torch::zeros_like(x).masked_scatter(expert_mask, expert_output);
//                 auto prob_mask = top_probs.select(-1, (top_indices == i).nonzero().select(-1, 0).item<int64_t>());
//                 output += expanded_output * prob_mask.unsqueeze(-1);
//             }
//         }
//         return output;
//     }
//
//     torch::nn::Linear gate{nullptr};
//     torch::nn::ModuleList experts{torch::nn::ModuleList()};
//     int num_experts_;
// };
// TORCH_MODULE(MoELayer);
//
// // Transformer Decoder Layer with MoE
// struct TransformerDecoderLayerImpl : torch::nn::Module {
//     TransformerDecoderLayerImpl(int d_model, int nhead, int num_experts, int dim_feedforward, float dropout) {
//         self_attn = register_module("self_attn", torch::nn::MultiheadAttention(
//             torch::nn::MultiheadAttentionOptions(d_model, nhead).dropout(dropout)));
//         moe = register_module("moe", MoELayer(d_model, num_experts, dim_feedforward, dropout));
//         norm1 = register_module("norm1", torch::nn::LayerNorm(d_model));
//         norm2 = register_module("norm2", torch::nn::LayerNorm(d_model));
//         dropout1 = register_module("dropout1", torch::nn::Dropout(dropout));
//         dropout2 = register_module("dropout2", torch::nn::Dropout(dropout));
//     }
//
//     torch::Tensor forward(torch::Tensor tgt, torch::Tensor tgt_mask = {}) {
//         tgt = norm1->forward(tgt); // Pre-LN
//         auto attn_out = self_attn->forward(tgt, tgt, tgt, {}, tgt_mask);
//         tgt = tgt + dropout1->forward(std::get<0>(attn_out));
//         tgt = norm2->forward(tgt); // Pre-LN
//         auto moe_out = moe->forward(tgt);
//         tgt = tgt + dropout2->forward(moe_out);
//         return tgt;
//     }
//
//     torch::nn::MultiheadAttention self_attn{nullptr};
//     MoELayer moe{nullptr};
//     torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
//     torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr};
// };
// TORCH_MODULE(TransformerDecoderLayer);
//
// // Transformer Decoder
// struct TransformerDecoderImpl : torch::nn::Module {
//     TransformerDecoderImpl(int d_model, int nhead, int num_layers, int num_experts, int dim_feedforward, float dropout) {
//         for (int i = 0; i < num_layers; ++i) {
//             layers->push_back(TransformerDecoderLayer(d_model, nhead, num_experts, dim_feedforward, dropout));
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
// // Simplified Grok-1-like Model
// struct Grok1Impl : torch::nn::Module {
//     Grok1Impl(int vocab_size, int d_model = 768, int nhead = 12, int num_layers = 12, int num_experts = 8, int dim_feedforward = 3072, float dropout = 0.1) {
//         token_embedding = register_module("token_embedding", torch::nn::Embedding(vocab_size, d_model));
//         positional_encoding = register_parameter("positional_encoding",
//             torch::randn({1, 2048, d_model}));
//         decoder = register_module("decoder", TransformerDecoder(d_model, nhead, num_layers, num_experts, dim_feedforward, dropout));
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
// TORCH_MODULE(Grok1);
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
//         const int d_model = 768;
//         const int nhead = 12;
//         const int num_layers = 12;
//         const int num_experts = 8;
//         const int dim_feedforward = 3072;
//         const float dropout = 0.1;
//         const int max_seq_length = 512;
//
//         // Load vocabulary
//         Vocabulary vocab("./data/vocab.txt");
//
//         // Initialize model
//         Grok1 model(vocab.vocab_size(), d_model, nhead, num_layers, num_experts, dim_feedforward, dropout);
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
//                 torch::save(model, "grok1_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: grok1_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "grok1_final.pt");
//         std::cout << "Saved final model: grok1_final.pt" << std::endl;
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
//
//     return 0;
// }

//Grok V2

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
// // Expert Feedforward Network
// struct ExpertFFNImpl : torch::nn::Module {
//     ExpertFFNImpl(int d_model, int dim_feedforward, float dropout) {
//         linear1 = register_module("linear1", torch::nn::Linear(d_model, dim_feedforward));
//         dropout1 = register_module("dropout1", torch::nn::Dropout(dropout));
//         linear2 = register_module("linear2", torch::nn::Linear(dim_feedforward, d_model));
//         dropout2 = register_module("dropout2", torch::nn::Dropout(dropout));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::swish(linear1->forward(x)); // Swish for Grok-2-like activation
//         x = dropout1->forward(x);
//         x = linear2->forward(x);
//         x = dropout2->forward(x);
//         return x;
//     }
//
//     torch::nn::Linear linear1{nullptr}, linear2{nullptr};
//     torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr};
// };
// TORCH_MODULE(ExpertFFN);
//
// // MoE Layer
// struct MoELayerImpl : torch::nn::Module {
//     MoELayerImpl(int d_model, int num_experts, int dim_feedforward, float dropout) : num_experts_(num_experts) {
//         gate = register_module("gate", torch::nn::Linear(d_model, num_experts));
//         for (int i = 0; i < num_experts; ++i) {
//             experts->push_back(ExpertFFN(d_model, dim_feedforward, dropout));
//             register_module("expert_" + std::to_string(i), experts->back());
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto gate_logits = gate->forward(x); // [batch, seq_len, num_experts]
//         auto gate_probs = torch::softmax(gate_logits, -1);
//         auto [top_probs, top_indices] = gate_probs.topk(2, -1, true, true); // Top-2 experts
//         top_probs = torch::softmax(top_probs, -1); // Normalize top-2
//
//         torch::Tensor output = torch::zeros_like(x);
//         for (int i = 0; i < num_experts_; ++i) {
//             auto expert_mask = (top_indices == i).any(-1, true); // [batch, seq_len, 1]
//             if (expert_mask.sum().item<int64_t>() > 0) {
//                 auto expert_input = x.masked_select(expert_mask).view({-1, x.size(-1)});
//                 auto expert_output = experts->at(i)->forward(expert_input);
//                 auto expanded_output = torch::zeros_like(x).masked_scatter(expert_mask, expert_output);
//                 auto prob_mask = top_probs.select(-1, (top_indices == i).nonzero().select(-1, 0).item<int64_t>());
//                 output += expanded_output * prob_mask.unsqueeze(-1);
//             }
//         }
//         return output;
//     }
//
//     torch::nn::Linear gate{nullptr};
//     torch::nn::ModuleList experts{torch::nn::ModuleList()};
//     int num_experts_;
// };
// TORCH_MODULE(MoELayer);
//
// // Transformer Decoder Layer with MoE
// struct TransformerDecoderLayerImpl : torch::nn::Module {
//     TransformerDecoderLayerImpl(int d_model, int nhead, int num_experts, int dim_feedforward, float dropout) {
//         self_attn = register_module("self_attn", torch::nn::MultiheadAttention(
//             torch::nn::MultiheadAttentionOptions(d_model, nhead).dropout(dropout)));
//         moe = register_module("moe", MoELayer(d_model, num_experts, dim_feedforward, dropout));
//         norm1 = register_module("norm1", torch::nn::LayerNorm(d_model));
//         norm2 = register_module("norm2", torch::nn::LayerNorm(d_model));
//         dropout1 = register_module("dropout1", torch::nn::Dropout(dropout));
//         dropout2 = register_module("dropout2", torch::nn::Dropout(dropout));
//     }
//
//     torch::Tensor forward(torch::Tensor tgt, torch::Tensor tgt_mask = {}) {
//         tgt = norm1->forward(tgt); // Pre-LN
//         auto attn_out = self_attn->forward(tgt, tgt, tgt, {}, tgt_mask);
//         tgt = tgt + dropout1->forward(std::get<0>(attn_out));
//         tgt = norm2->forward(tgt); // Pre-LN
//         auto moe_out = moe->forward(tgt);
//         tgt = tgt + dropout2->forward(moe_out);
//         return tgt;
//     }
//
//     torch::nn::MultiheadAttention self_attn{nullptr};
//     MoELayer moe{nullptr};
//     torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
//     torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr};
// };
// TORCH_MODULE(TransformerDecoderLayer);
//
// // Transformer Decoder
// struct TransformerDecoderImpl : torch::nn::Module {
//     TransformerDecoderImpl(int d_model, int nhead, int num_layers, int num_experts, int dim_feedforward, float dropout) {
//         for (int i = 0; i < num_layers; ++i) {
//             layers->push_back(TransformerDecoderLayer(d_model, nhead, num_experts, dim_feedforward, dropout));
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
// // Simplified Grok-2-like Model
// struct Grok2Impl : torch::nn::Module {
//     Grok2Impl(int vocab_size, int d_model = 1024, int nhead = 16, int num_layers = 24, int num_experts = 16, int dim_feedforward = 4096, float dropout = 0.1) {
//         token_embedding = register_module("token_embedding", torch::nn::Embedding(vocab_size, d_model));
//         positional_encoding = register_parameter("positional_encoding",
//             torch::randn({1, 8192, d_model})); // Larger context for Grok-2
//         decoder = register_module("decoder", TransformerDecoder(d_model, nhead, num_layers, num_experts, dim_feedforward, dropout));
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
// TORCH_MODULE(Grok2);
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
//         const float learning_rate = 0.00005; // Lower LR for stability
//         const int d_model = 1024; // Increased for Grok-2
//         const int nhead = 16;
//         const int num_layers = 24; // Deeper model
//         const int num_experts = 16; // More experts
//         const int dim_feedforward = 4096;
//         const float dropout = 0.1;
//         const int max_seq_length = 1024; // Longer sequences
//
//         // Load vocabulary
//         Vocabulary vocab("./data/vocab.txt");
//
//         // Initialize model
//         Grok2 model(vocab.vocab_size(), d_model, nhead, num_layers, num_experts, dim_feedforward, dropout);
//         model->to(device);
//
//         // Optimizer
//         torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(learning_rate).weight_decay(0.01));
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
//                 torch::save(model, "grok2_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: grok2_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "grok2_final.pt");
//         std::cout << "Saved final model: grok2_final.pt" << std::endl;
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
    Grok::Grok(int num_classes, int in_channels)
    {
    }

    Grok::Grok(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void Grok::reset()
    {
    }

    auto Grok::forward(std::initializer_list<std::any> tensors) -> std::any
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
