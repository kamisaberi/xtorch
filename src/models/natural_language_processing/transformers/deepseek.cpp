#include "include/models/natural_language_processing/transformers/deepseek.h"


using namespace std;
//DeepSeekV2

#include <torch/torch.h>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <map>
#include <random>
#include <sstream>

// Vocabulary Class
class Vocabulary {
public:
    Vocabulary(const std::string& vocab_file) {
        std::ifstream file(vocab_file);
        std::string token;
        int idx = 0;
        while (std::getline(file, token)) {
            token_to_idx_[token] = idx;
            idx_to_token_[idx] = token;
            idx++;
        }
        vocab_size_ = idx;
    }

    int token_to_idx(const std::string& token) const {
        auto it = token_to_idx_.find(token);
        return it != token_to_idx_.end() ? it->second : token_to_idx_.at("<unk>");
    }

    std::string idx_to_token(int idx) const {
        auto it = idx_to_token_.find(idx);
        return it != idx_to_token_.end() ? it->second : "<unk>";
    }

    int vocab_size() const { return vocab_size_; }

private:
    std::map<std::string, int> token_to_idx_;
    std::map<int, std::string> idx_to_token_;
    int vocab_size_;
};

// Custom Prompt-Response Dataset
class PromptResponseDataset : public torch::data::Dataset<PromptResponseDataset> {
public:
    PromptResponseDataset(const std::string& prompt_dir, const std::string& response_dir, const Vocabulary& vocab, int max_seq_length)
        : vocab_(vocab), max_seq_length_(max_seq_length) {
        for (const auto& entry : std::filesystem::directory_iterator(prompt_dir)) {
            if (entry.path().extension() == ".txt") {
                prompt_paths_.push_back(entry.path().string());
                std::string response_path = response_dir + "/" + entry.path().filename().string();
                response_paths_.push_back(response_path);
            }
        }
    }

    torch::data::Example<> get(size_t index) override {
        // Load prompt
        std::ifstream prompt_file(prompt_paths_[index]);
        if (!prompt_file.is_open()) {
            throw std::runtime_error("Failed to open prompt file: " + prompt_paths_[index]);
        }
        std::string prompt;
        std::getline(prompt_file, prompt);

        // Load response
        std::ifstream response_file(response_paths_[index]);
        if (!response_file.is_open()) {
            throw std::runtime_error("Failed to open response file: " + response_paths_[index]);
        }
        std::string response;
        std::string line;
        while (std::getline(response_file, line)) {
            response += line + " ";
        }

        // Tokenize prompt and response
        std::vector<std::string> tokens;
        tokens.push_back("<sos>");
        std::istringstream prompt_iss(prompt);
        std::string token;
        while (prompt_iss >> token && tokens.size() < max_seq_length_ / 2) {
            tokens.push_back(token);
        }
        tokens.push_back("<sep>");
        std::istringstream response_iss(response);
        while (response_iss >> token && tokens.size() < max_seq_length_ - 1) {
            tokens.push_back(token);
        }
        tokens.push_back("<eos>");
        while (tokens.size() < max_seq_length_) {
            tokens.push_back("<pad>");
        }

        // Convert to indices
        std::vector<int64_t> token_indices;
        for (const auto& t : tokens) {
            token_indices.push_back(vocab_.token_to_idx(t));
        }
        torch::Tensor input_tensor = torch::tensor(token_indices, torch::kInt64);

        // Target is shifted right (exclude <sos>, predict up to <eos>)
        std::vector<int64_t> target_indices(token_indices.begin() + 1, token_indices.end());
        target_indices.pop_back();
        target_indices.push_back(vocab_.token_to_idx("<pad>"));
        torch::Tensor target_tensor = torch::tensor(target_indices, torch::kInt64);

        return {input_tensor, target_tensor};
    }

    torch::optional<size_t> size() const override {
        return prompt_paths_.size();
    }

private:
    std::vector<std::string> prompt_paths_, response_paths_;
    const Vocabulary& vocab_;
    int max_seq_length_;
};

// MoE Expert (Feedforward Network)
struct ExpertImpl : torch::nn::Module {
    ExpertImpl(int d_model, int dim_feedforward) {
        linear1 = register_module("linear1", torch::nn::Linear(d_model, dim_feedforward));
        linear2 = register_module("linear2", torch::nn::Linear(dim_feedforward, d_model));
        dropout = register_module("dropout", torch::nn::Dropout(0.1));
    }

    torch::Tensor forward(torch::Tensor x) {
        return linear2->forward(dropout->forward(torch::relu(linear1->forward(x))));
    }

    torch::nn::Linear linear1{nullptr}, linear2{nullptr};
    torch::nn::Dropout dropout{nullptr};
};
TORCH_MODULE(Expert);

// MoE Layer
struct MoELayerImpl : torch::nn::Module {
    MoELayerImpl(int d_model, int dim_feedforward, int num_experts, int top_k) : top_k_(top_k) {
        gating = register_module("gating", torch::nn::Linear(d_model, num_experts));
        for (int i = 0; i < num_experts; ++i) {
            experts->push_back(Expert(d_model, dim_feedforward));
            register_module("expert_" + std::to_string(i), experts->back());
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        auto gate_logits = gating->forward(x); // [batch, seq_len, num_experts]
        auto topk = torch::topk(gate_logits, top_k_, /*dim*/ -1);
        auto weights = torch::softmax(std::get<0>(topk), -1); // [batch, seq_len, top_k]
        auto indices = std::get<1>(topk); // [batch, seq_len, top_k]

        torch::Tensor output = torch::zeros_like(x);
        for (int i = 0; i < top_k_; ++i) {
            auto w = weights.select(-1, i).unsqueeze(-1); // [batch, seq_len, 1]
            auto idx = indices.select(-1, i); // [batch, seq_len]
            torch::Tensor expert_output = torch::zeros_like(x);
            for (int b = 0; b < x.size(0); ++b) {
                for (int s = 0; s < x.size(1); ++s) {
                    int e = idx[b][s].item<int>();
                    auto input_slice = x.select(0, b).select(0, s).unsqueeze(0);
                    auto out_slice = experts[e]->forward(input_slice);
                    expert_output[b][s] = out_slice.squeeze(0);
                }
            }
            output += w * expert_output;
        }
        return output;
    }

    torch::nn::Linear gating{nullptr};
    torch::nn::ModuleList experts{torch::nn::ModuleList()};
    int top_k_;
};
TORCH_MODULE(MoELayer);

// Transformer Encoder Layer
struct TransformerEncoderLayerImpl : torch::nn::Module {
    TransformerEncoderLayerImpl(int d_model, int nhead, int dim_feedforward, int num_experts, int top_k, float dropout) {
        self_attn = register_module("self_attn", torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(d_model, nhead).dropout(dropout)));
        moe = register_module("moe", MoELayer(d_model, dim_feedforward, num_experts, top_k));
        norm1 = register_module("norm1", torch::nn::LayerNorm(d_model));
        norm2 = register_module("norm2", torch::nn::LayerNorm(d_model));
        dropout1 = register_module("dropout1", torch::nn::Dropout(dropout));
        dropout2 = register_module("dropout2", torch::nn::Dropout(dropout));
    }

    torch::Tensor forward(torch::Tensor src, torch::Tensor src_mask = {}) {
        auto attn_out = self_attn->forward(src, src, src, {}, src_mask);
        src = norm1->forward(src + dropout1->forward(std::get<0>(attn_out)));
        auto moe_out = moe->forward(src);
        src = norm2->forward(src + dropout2->forward(moe_out));
        return src;
    }

    torch::nn::MultiheadAttention self_attn{nullptr};
    MoELayer moe{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr};
};
TORCH_MODULE(TransformerEncoderLayer);

// Transformer Encoder
struct TransformerEncoderImpl : torch::nn::Module {
    TransformerEncoderImpl(int d_model, int nhead, int num_layers, int dim_feedforward, int num_experts, int top_k, float dropout) {
        for (int i = 0; i < num_layers; ++i) {
            layers->push_back(TransformerEncoderLayer(d_model, nhead, dim_feedforward, num_experts, top_k, dropout));
            register_module("layer_" + std::to_string(i), layers->back());
        }
    }

    torch::Tensor forward(torch::Tensor src, torch::Tensor src_mask = {}) {
        torch::Tensor output = src;
        for (auto& layer : *layers) {
            output = layer->forward(output, src_mask);
        }
        return output;
    }

    torch::nn::ModuleList layers{torch::nn::ModuleList()};
};
TORCH_MODULE(TransformerEncoder);

// Simplified DeepSeek-V2-like Model
struct DeepSeekV2Impl : torch::nn::Module {
    DeepSeekV2Impl(int vocab_size, int d_model = 256, int nhead = 4, int num_layers = 2, int dim_feedforward = 512, int num_experts = 4, int top_k = 2, float dropout = 0.1) {
        token_embedding = register_module("token_embedding", torch::nn::Embedding(vocab_size, d_model));
        positional_encoding = register_parameter("positional_encoding",
            torch::randn({1, 128, d_model}));
        encoder = register_module("encoder", TransformerEncoder(d_model, nhead, num_layers, dim_feedforward, num_experts, top_k, dropout));
        output_layer = register_module("output_layer", torch::nn::Linear(d_model, vocab_size));
    }

    torch::Tensor forward(torch::Tensor input) {
        auto features = token_embedding->forward(input); // [batch, seq_len, d_model]
        features = features + positional_encoding.slice(1, 0, input.size(1));
        auto causal_mask = torch::triu(torch::ones({input.size(1), input.size(1)}), 1).to(torch::kBool);
        auto encoded = encoder->forward(features, causal_mask); // [batch, seq_len, d_model]
        return output_layer->forward(encoded); // [batch, seq_len, vocab_size]
    }

    torch::nn::Embedding token_embedding{nullptr};
    torch::Tensor positional_encoding;
    TransformerEncoder encoder{nullptr};
    torch::nn::Linear output_layer{nullptr};
};
TORCH_MODULE(DeepSeekV2);

int main() {
    try {
        // Set device
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

        // Hyperparameters
        const int batch_size = 16;
        const int num_epochs = 10;
        const float learning_rate = 0.001;
        const int d_model = 256;
        const int nhead = 4;
        const int num_layers = 2;
        const int dim_feedforward = 512;
        const int num_experts = 4;
        const int top_k = 2;
        const float dropout = 0.1;
        const int max_seq_length = 128;

        // Load vocabulary
        Vocabulary vocab("./data/vocab.txt");

        // Initialize model
        DeepSeekV2 model(vocab.vocab_size(), d_model, nhead, num_layers, dim_feedforward, num_experts, top_k, dropout);
        model->to(device);

        // Optimizer
        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

        // Dataset and DataLoader
        auto dataset = PromptResponseDataset("./data/prompts", "./data/responses", vocab, max_seq_length)
            .map(torch::data::transforms::Stack<>());
        auto data_loader = torch::data::make_data_loader(
            std::move(dataset),
            torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));

        // Training loop
        model->train();
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            float total_loss = 0.0;
            int batch_count = 0;
            for (auto& batch : *data_loader) {
                auto inputs = batch.data.to(device);
                auto targets = batch.target.to(device);

                optimizer.zero_grad();
                auto outputs = model->forward(inputs);
                auto loss = torch::nn::functional::cross_entropy(
                    outputs.view({-1, vocab.vocab_size()}),
                    targets.view({-1})
                );
                loss.backward();
                optimizer.step();

                total_loss += loss.item<float>();
                batch_count++;
            }

            float avg_loss = total_loss / batch_count;
            std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "] "
                      << "Loss: " << avg_loss << std::endl;

            // Save checkpoint every 5 epochs
            if ((epoch + 1) % 5 == 0) {
                torch::save(model, "deepseek_v2_epoch_" + std::to_string(epoch + 1) + ".pt");
                std::cout << "Saved checkpoint: deepseek_v2_epoch_" << epoch + 1 << ".pt" << std::endl;
            }
        }

        // Save final model
        torch::save(model, "deepseek_v2_final.pt");
        std::cout << "Saved final model: deepseek_v2_final.pt" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

namespace xt::models
{
    DeepSeek::DeepSeek(int num_classes, int in_channels)
    {
    }

    DeepSeek::DeepSeek(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DeepSeek::reset()
    {
    }

    auto DeepSeek::forward(std::initializer_list<std::any> tensors) -> std::any
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
