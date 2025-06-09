#include "include/models/time_series_and_sequence/times_former.h"


using namespace std;


#include <torch/torch.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <iostream>
#include <cmath>

// ProbSparse Self-Attention Module
struct ProbSparseAttention : torch::nn::Module {
    int d_model, n_heads, factor;
    torch::nn::Linear q_linear{nullptr}, k_linear{nullptr}, v_linear{nullptr};
    torch::nn::Linear out_linear{nullptr};
    torch::nn::Dropout dropout{nullptr};

    ProbSparseAttention(int d_model_, int n_heads_, int factor_, float dropout_p = 0.1)
        : d_model(d_model_), n_heads(n_heads_), factor(factor_) {
        q_linear = register_module("q_linear", torch::nn::Linear(d_model, d_model));
        k_linear = register_module("k_linear", torch::nn::Linear(d_model, d_model));
        v_linear = register_module("v_linear", torch::nn::Linear(d_model, d_model));
        out_linear = register_module("out_linear", torch::nn::Linear(d_model, d_model));
        dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
    }

    torch::Tensor forward(torch::Tensor query, torch::Tensor key, torch::Tensor value) {
        int64_t batch_size = query.size(0), seq_len = query.size(1);
        int64_t head_dim = d_model / n_heads;

        // Linear projections
        auto q = q_linear->forward(query).view({batch_size, seq_len, n_heads, head_dim}).transpose(1, 2);
        auto k = k_linear->forward(key).view({batch_size, seq_len, n_heads, head_dim}).transpose(1, 2);
        auto v = v_linear->forward(value).view({batch_size, seq_len, n_heads, head_dim}).transpose(1, 2);

        // ProbSparse attention scores
        auto scores = torch::bmm(q, k.transpose(2, 3)) / std::sqrt(static_cast<float>(head_dim));
        auto max_scores = scores.max(-1).values.unsqueeze(-1);
        auto sparsity_scores = torch::logsumexp(scores - max_scores, -1) - max_scores.squeeze(-1);
        auto topk = sparsity_scores.topk(factor * static_cast<int>(std::log(seq_len)), -1, true, true);
        auto mask = torch::zeros_like(scores).to(query.device());
        mask.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), topk.indices}, torch::ones_like(topk.values));
        scores = scores * mask - 1e9 * (1 - mask);

        // Softmax and attention output
        auto attn = torch::softmax(scores, -1);
        attn = dropout->forward(attn);
        auto out = torch::bmm(attn, v);
        out = out.transpose(1, 2).contiguous().view({batch_size, seq_len, d_model});
        return out_linear->forward(out);
    }
};

// Informer Encoder Layer with Distilling
struct InformerEncoderLayer : torch::nn::Module {
    ProbSparseAttention attention{nullptr};
    torch::nn::Linear ff_linear1{nullptr}, ff_linear2{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::nn::Conv1d distil_conv{nullptr};

    InformerEncoderLayer(int d_model, int n_heads, int d_ff, int factor, float dropout_p = 0.1)
        : attention(ProbSparseAttention(d_model, n_heads, factor, dropout_p)),
          norm1(register_module("norm1", torch::nn::LayerNorm(d_model))),
          norm2(register_module("norm2", torch::nn::LayerNorm(d_model))),
          ff_linear1(register_module("ff_linear1", torch::nn::Linear(d_model, d_ff))),
          ff_linear2(register_module("ff_linear2", torch::nn::Linear(d_ff, d_model))),
          dropout(register_module("dropout", torch::nn::Dropout(dropout_p))),
          distil_conv(register_module("distil_conv", torch::nn::Conv1d(d_model, d_model, 3, torch::nn::Conv1dOptions().stride(2).padding(1)))) {
    }

    torch::Tensor forward(torch::Tensor x) {
        // Attention
        auto attn_out = attention.forward(x, x, x);
        x = norm1->forward(x + dropout->forward(attn_out));

        // Feed-forward
        auto ff_out = ff_linear2->forward(torch::relu(ff_linear1->forward(x)));
        x = norm2->forward(x + dropout->forward(ff_out));

        // Distilling
        x = x.transpose(1, 2);
        x = distil_conv->forward(x);
        x = x.transpose(1, 2);
        return x;
    }
};

// Informer Decoder Layer
struct InformerDecoderLayer : torch::nn::Module {
    ProbSparseAttention self_attention{nullptr}, cross_attention{nullptr};
    torch::nn::Linear ff_linear1{nullptr}, ff_linear2{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr}, norm3{nullptr};
    torch::nn::Dropout dropout{nullptr};

    InformerDecoderLayer(int d_model, int n_heads, int d_ff, int factor, float dropout_p = 0.1)
        : self_attention(ProbSparseAttention(d_model, n_heads, factor, dropout_p)),
          cross_attention(ProbSparseAttention(d_model, n_heads, factor, dropout_p)),
          norm1(register_module("norm1", torch::nn::LayerNorm(d_model))),
          norm2(register_module("norm2", torch::nn::LayerNorm(d_model))),
          norm3(register_module("norm3", torch::nn::LayerNorm(d_model))),
          ff_linear1(register_module("ff_linear1", torch::nn::Linear(d_model, d_ff))),
          ff_linear2(register_module("ff_linear2", torch::nn::Linear(d_ff, d_model))),
          dropout(register_module("dropout", torch::nn::Dropout(dropout_p))) {
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor enc_out) {
        // Self-attention
        auto self_attn_out = self_attention.forward(x, x, x);
        x = norm1->forward(x + dropout->forward(self_attn_out));

        // Cross-attention
        auto cross_attn_out = cross_attention.forward(x, enc_out, enc_out);
        x = norm2->forward(x + dropout->forward(cross_attn_out));

        // Feed-forward
        auto ff_out = ff_linear2->forward(torch::relu(ff_linear1->forward(x)));
        x = norm3->forward(x + dropout->forward(ff_out));
        return x;
    }
};

// TimeSInformer (Informer-inspired) Model
struct TimeSInformer : torch::nn::Module {
    torch::nn::Linear input_projection{nullptr};
    torch::nn::Embedding token_embedding{nullptr};
    std::vector<InformerEncoderLayer> encoder_layers;
    std::vector<InformerDecoderLayer> decoder_layers;
    torch::nn::Linear output_projection{nullptr};
    int seq_len, pred_len, d_model;

    TimeSInformer(int seq_len_, int pred_len_, int d_model_, int n_heads, int e_layers, int d_layers, int d_ff, int factor, int max_tokens = 1000)
        : seq_len(seq_len_), pred_len(pred_len_), d_model(d_model_) {
        input_projection = register_module("input_projection", torch::nn::Linear(1, d_model));
        token_embedding = register_module("token_embedding", torch::nn::Embedding(max_tokens, d_model));
        for (int i = 0; i < e_layers; ++i) {
            encoder_layers.push_back(register_module("enc_layer_" + std::to_string(i),
                                                    InformerEncoderLayer(d_model, n_heads, d_ff, factor)));
        }
        for (int i = 0; i < d_layers; ++i) {
            decoder_layers.push_back(register_module("dec_layer_" + std::to_string(i),
                                                    InformerDecoderLayer(d_model, n_heads, d_ff, factor)));
        }
        output_projection = register_module("output_projection", torch::nn::Linear(d_model, 1));
    }

    torch::Tensor forward(torch::Tensor src) {
        // Encoder
        auto src_emb = input_projection->forward(src);
        auto enc_out = src_emb;
        for (auto& layer : encoder_layers) {
            enc_out = layer.forward(enc_out);
        }

        // Decoder
        auto tgt_tokens = torch::arange(pred_len).unsqueeze(0).repeat({src.size(0), 1}).to(src.device());
        auto tgt_emb = token_embedding->forward(tgt_tokens);
        auto dec_out = tgt_emb;
        for (auto& layer : decoder_layers) {
            dec_out = layer.forward(dec_out, enc_out);
        }

        // Output projection
        return output_projection->forward(dec_out);
    }
};

// Time-Series Dataset
struct TimeSeriesDataset {
    std::vector<std::vector<float>> sequences;
    float mean = 0.0, std = 1.0;
    int seq_len, pred_len;

    TimeSeriesDataset(const std::string& filename, int seq_len_, int pred_len_) : seq_len(seq_len_), pred_len(pred_len_) {
        std::ifstream file(filename);
        std::string line;
        std::vector<float> data;
        while (std::getline(file, line)) {
            try {
                data.push_back(std::stof(line));
            } catch (...) {
                continue;
            }
        }

        // Normalize data
        float sum = std::accumulate(data.begin(), data.end(), 0.0);
        mean = sum / data.size();
        float sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
        std = std::sqrt(sq_sum / data.size() - mean * mean);
        for (auto& x : data) {
            x = (x - mean) / (std + 1e-8);
        }

        // Create sequences
        for (size_t i = 0; i + seq_len + pred_len <= data.size(); ++i) {
            std::vector<float> seq(data.begin() + i, data.begin() + i + seq_len + pred_len);
            sequences.push_back(seq);
        }
    }

    std::tuple<torch::Tensor, torch::Tensor> get_batch(size_t idx, size_t batch_size) {
        std::vector<torch::Tensor> src_batch, tgt_batch;
        for (size_t i = idx; i < std::min(idx + batch_size, sequences.size()); ++i) {
            auto& seq = sequences[i];
            std::vector<float> src(seq.begin(), seq.begin() + seq_len);
            std::vector<float> tgt(seq.begin() + seq_len, seq.end());
            src_batch.push_back(torch::tensor(src).unsqueeze(-1));
            tgt_batch.push_back(torch::tensor(tgt).unsqueeze(-1));
        }
        return {torch::stack(src_batch), torch::stack(tgt_batch)};
    }
};

int main() {
    torch::manual_seed(42);
    std::random_device rd;
    std::mt19937 gen(rd());

    // Load dataset
    TimeSeriesDataset dataset("timeseries_data.csv", 96, 48);
    if (dataset.sequences.empty()) {
        std::cerr << "No valid sequences found in dataset." << std::endl;
        return -1;
    }

    // Initialize model
    TimeSInformer model(
        96, 48, 64, 4, 2, 2, 256, 5 // seq_len, pred_len, d_model, n_heads, e_layers, d_layers, d_ff, factor
    );
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    model.to(device);

    // Optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));

    // Training loop
    const int epochs = 20;
    const size_t batch_size = 32;
    auto criterion = torch::nn::MSELoss();
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0;
        size_t num_batches = 0;
        model.train();

        for (size_t i = 0; i < dataset.sequences.size(); i += batch_size) {
            auto [src, tgt] = dataset.get_batch(i, batch_size);
            src = src.to(device);
            tgt = tgt.to(device);

            optimizer.zero_grad();
            auto output = model.forward(src);
            auto loss = criterion(output, tgt);
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            optimizer.step();

            total_loss += loss.item<float>();
            num_batches++;
        }

        std::cout << "Epoch: " << epoch + 1 << ", Loss: " << total_loss / num_batches << std::endl;
    }

    // Save model
    torch::save(model, "timesinformer_model.pt");

    return 0;
}


namespace xt::models
{
    TimeSInformer::TimeSInformer(int num_classes, int in_channels)
    {
    }

    TimeSInformer::TimeSInformer(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void TimeSInformer::reset()
    {
    }

    auto TimeSInformer::forward(std::initializer_list<std::any> tensors) -> std::any
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
