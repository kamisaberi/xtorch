#include <models/natural_language_processing/transformers/gpt.h>


using namespace std;

//GPT V1


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
//         // Load prompt
//         std::ifstream prompt_file(prompt_paths_[index]);
//         if (!prompt_file.is_open()) {
//             throw std::runtime_error("Failed to open prompt file: " + prompt_paths_[index]);
//         }
//         std::string prompt;
//         std::getline(prompt_file, prompt);
//
//         // Load response
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
//         // Tokenize prompt and response
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
//         // Convert to indices
//         std::vector<int64_t> token_indices;
//         for (const auto& t : tokens) {
//             token_indices.push_back(vocab_.token_to_idx(t));
//         }
//         torch::Tensor input_tensor = torch::tensor(token_indices, torch::kInt64);
//
//         // Target is shifted right (exclude <sos>, predict up to <eos>)
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
// // Transformer Decoder Layer
// struct TransformerDecoderLayerImpl : torch::nn::Module {
//     TransformerDecoderLayerImpl(int d_model, int nhead, int dim_feedforward, float dropout) {
//         self_attn = register_module("self_attn", torch::nn::MultiheadAttention(
//             torch::nn::MultiheadAttentionOptions(d_model, nhead).dropout(dropout)));
//         linear1 = register_module("linear1", torch::nn::Linear(d_model, dim_feedforward));
//         dropout1 = register_module("dropout1", torch::nn::Dropout(dropout));
//         linear2 = register_module("linear2", torch::nn::Linear(dim_feedforward, d_model));
//         norm1 = register_module("norm1", torch::nn::LayerNorm(d_model));
//         norm2 = register_module("norm2", torch::nn::LayerNorm(d_model));
//         dropout2 = register_module("dropout2", torch::nn::Dropout(dropout));
//     }
//
//     torch::Tensor forward(torch::Tensor tgt, torch::Tensor tgt_mask = {}) {
//         auto attn_out = self_attn->forward(tgt, tgt, tgt, {}, tgt_mask);
//         tgt = norm1->forward(tgt + dropout1->forward(std::get<0>(attn_out)));
//         auto ff_out = linear2->forward(torch::gelu(linear1->forward(tgt)));
//         tgt = norm2->forward(tgt + dropout2->forward(ff_out));
//         return tgt;
//     }
//
//     torch::nn::MultiheadAttention self_attn{nullptr};
//     torch::nn::Linear linear1{nullptr}, linear2{nullptr};
//     torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr};
//     torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
// };
// TORCH_MODULE(TransformerDecoderLayer);
//
// // Transformer Decoder
// struct TransformerDecoderImpl : torch::nn::Module {
//     TransformerDecoderImpl(int d_model, int nhead, int num_layers, int dim_feedforward, float dropout) {
//         for (int i = 0; i < num_layers; ++i) {
//             layers->push_back(TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout));
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
// // Simplified GPT-1-like Model
// struct GPT1Impl : torch::nn::Module {
//     GPT1Impl(int vocab_size, int d_model = 256, int nhead = 4, int num_layers = 2, int dim_feedforward = 1024, float dropout = 0.1) {
//         token_embedding = register_module("token_embedding", torch::nn::Embedding(vocab_size, d_model));
//         positional_encoding = register_parameter("positional_encoding",
//             torch::randn({1, 512, d_model}));
//         decoder = register_module("decoder", TransformerDecoder(d_model, nhead, num_layers, dim_feedforward, dropout));
//         output_layer = register_module("output_layer", torch::nn::Linear(d_model, vocab_size));
//     }
//
//     torch::Tensor forward(torch::Tensor input) {
//         auto features = token_embedding->forward(input); // [batch, seq_len, d_model]
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
// TORCH_MODULE(GPT1);
//
// int main() {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Hyperparameters
//         const int batch_size = 16;
//         const int num_epochs = 10;
//         const float learning_rate = 0.00025;
//         const int d_model = 256;
//         const int nhead = 4;
//         const int num_layers = 2;
//         const int dim_feedforward = 1024;
//         const float dropout = 0.1;
//         const int max_seq_length = 128;
//
//         // Load vocabulary
//         Vocabulary vocab("./data/vocab.txt");
//
//         // Initialize model
//         GPT1 model(vocab.vocab_size(), d_model, nhead, num_layers, dim_feedforward, dropout);
//         model->to(device);
//
//         // Optimizer
//         torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(0.01));
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
//                 torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0); // Gradient clipping
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
//                 torch::save(model, "gpt1_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: gpt1_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "gpt1_final.pt");
//         std::cout << "Saved final model: gpt1_final.pt" << std::endl;
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
//
//     return 0;
// }

//GPT V2


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
//         // Load prompt
//         std::ifstream prompt_file(prompt_paths_[index]);
//         if (!prompt_file.is_open()) {
//             throw std::runtime_error("Failed to open prompt file: " + prompt_paths_[index]);
//         }
//         std::string prompt;
//         std::getline(prompt_file, prompt);
//
//         // Load response
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
//         // Tokenize prompt and response
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
//         // Convert to indices
//         std::vector<int64_t> token_indices;
//         for (const auto& t : tokens) {
//             token_indices.push_back(vocab_.token_to_idx(t));
//         }
//         torch::Tensor input_tensor = torch::tensor(token_indices, torch::kInt64);
//
//         // Target is shifted right (exclude <sos>, predict up to <eos>)
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
// // Transformer Decoder Layer
// struct TransformerDecoderLayerImpl : torch::nn::Module {
//     TransformerDecoderLayerImpl(int d_model, int nhead, int dim_feedforward, float dropout) {
//         self_attn = register_module("self_attn", torch::nn::MultiheadAttention(
//             torch::nn::MultiheadAttentionOptions(d_model, nhead).dropout(dropout)));
//         linear1 = register_module("linear1", torch::nn::Linear(d_model, dim_feedforward));
//         dropout1 = register_module("dropout1", torch::nn::Dropout(dropout));
//         linear2 = register_module("linear2", torch::nn::Linear(dim_feedforward, d_model));
//         norm1 = register_module("norm1", torch::nn::LayerNorm(d_model));
//         norm2 = register_module("norm2", torch::nn::LayerNorm(d_model));
//         dropout2 = register_module("dropout2", torch::nn::Dropout(dropout));
//     }
//
//     torch::Tensor forward(torch::Tensor tgt, torch::Tensor tgt_mask = {}) {
//         auto attn_out = self_attn->forward(tgt, tgt, tgt, {}, tgt_mask);
//         tgt = norm1->forward(tgt + dropout1->forward(std::get<0>(attn_out)));
//         auto ff_out = linear2->forward(torch::gelu(linear1->forward(tgt)));
//         tgt = norm2->forward(tgt + dropout2->forward(ff_out));
//         return tgt;
//     }
//
//     torch::nn::MultiheadAttention self_attn{nullptr};
//     torch::nn::Linear linear1{nullptr}, linear2{nullptr};
//     torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr};
//     torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
// };
// TORCH_MODULE(TransformerDecoderLayer);
//
// // Transformer Decoder
// struct TransformerDecoderImpl : torch::nn::Module {
//     TransformerDecoderImpl(int d_model, int nhead, int num_layers, int dim_feedforward, float dropout) {
//         for (int i = 0; i < num_layers; ++i) {
//             layers->push_back(TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout));
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
// // Simplified GPT-2-like Model
// struct GPT2Impl : torch::nn::Module {
//     GPT2Impl(int vocab_size, int d_model = 512, int nhead = 8, int num_layers = 6, int dim_feedforward = 2048, float dropout = 0.1) {
//         token_embedding = register_module("token_embedding", torch::nn::Embedding(vocab_size, d_model));
//         positional_encoding = register_parameter("positional_encoding",
//             torch::randn({1, 1024, d_model}));
//         decoder = register_module("decoder", TransformerDecoder(d_model, nhead, num_layers, dim_feedforward, dropout));
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
// TORCH_MODULE(GPT2);
//
// int main() {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Hyperparameters
//         const int batch_size = 16;
//         const int num_epochs = 10;
//         const float learning_rate = 0.0005;
//         const int d_model = 512;
//         const int nhead = 8;
//         const int num_layers = 6;
//         const int dim_feedforward = 2048;
//         const float dropout = 0.1;
//         const int max_seq_length = 256;
//
//         // Load vocabulary
//         Vocabulary vocab("./data/vocab.txt");
//
//         // Initialize model
//         GPT2 model(vocab.vocab_size(), d_model, nhead, num_layers, dim_feedforward, dropout);
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
//                 torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0); // Gradient clipping
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
//                 torch::save(model, "gpt2_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: gpt2_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "gpt2_final.pt");
//         std::cout << "Saved final model: gpt2_final.pt" << std::endl;
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
    GPT::GPT(int num_classes, int in_channels)
    {
    }

    GPT::GPT(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void GPT::reset()
    {
    }

    auto GPT::forward(std::initializer_list<std::any> tensors) -> std::any
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
