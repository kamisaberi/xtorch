#include "include/models/time_series_and_sequence/tcn.h"


using namespace std;


#include <torch/torch.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <iostream>
#include <cmath>

// TCN Block (Causal Dilated Convolution + Residual)
struct TCNBlock : torch::nn::Module {
    torch::nn::Conv1d conv1{nullptr}, conv2{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr};
    torch::nn::Conv1d residual_conv{nullptr};
    int dilation, channels;

    TCNBlock(int in_channels, int out_channels, int kernel_size, int dilation_, float dropout_p = 0.2)
        : dilation(dilation_), channels(out_channels) {
        conv1 = register_module("conv1", torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                                                           .stride(1)
                                                           .padding((kernel_size - 1) * dilation)
                                                           .dilation(dilation)));
        conv2 = register_module("conv2", torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channels, out_channels, kernel_size)
                                                           .stride(1)
                                                           .padding((kernel_size - 1) * dilation)
                                                           .dilation(dilation)));
        norm1 = register_module("norm1", torch::nn::LayerNorm(std::vector<int64_t>{out_channels}));
        norm2 = register_module("norm2", torch::nn::LayerNorm(std::vector<int64_t>{out_channels}));
        dropout1 = register_module("dropout1", torch::nn::Dropout(dropout_p));
        dropout2 = register_module("dropout2", torch::nn::Dropout(dropout_p));
        if (in_channels != out_channels) {
            residual_conv = register_module("residual_conv", torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, out_channels, 1)));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        // Input: [batch, in_channels, seq_len]
        auto out = conv1->forward(x); // [batch, out_channels, seq_len]
        out = torch::relu(out);
        out = norm1->forward(out.transpose(1, 2)).transpose(1, 2); // Normalize over channels
        out = dropout1->forward(out);

        out = conv2->forward(out); // [batch, out_channels, seq_len]
        out = torch::relu(out);
        out = norm2->forward(out.transpose(1, 2)).transpose(1, 2);
        out = dropout2->forward(out);

        // Residual connection
        auto residual = x;
        if (residual_conv) {
            residual = residual_conv->forward(x);
        }
        // Trim residual to match output length (due to causal padding)
        residual = residual.slice(2, (conv1->options.padding(0) + conv2->options.padding(0)));
        out = out + residual;

        return out; // [batch, out_channels, seq_len]
    }
};

// TCN Model
struct TCN : torch::nn::Module {
    std::vector<TCNBlock> blocks;
    torch::nn::Linear output_projection{nullptr};
    int seq_len, pred_len;

    TCN(int in_channels, int out_channels, int kernel_size, int num_layers, int seq_len_, int pred_len_)
        : seq_len(seq_len_), pred_len(pred_len_) {
        int dilation = 1;
        for (int i = 0; i < num_layers; ++i) {
            int in_ch = (i == 0) ? in_channels : out_channels;
            blocks.push_back(register_module("block_" + std::to_string(i),
                                            TCNBlock(in_ch, out_channels, kernel_size, dilation)));
            dilation *= 2; // Exponential dilation
        }
        output_projection = register_module("output_projection", torch::nn::Linear(out_channels, 1));
    }

    torch::Tensor forward(torch::Tensor x) {
        // Input: [batch, seq_len, in_channels]
        x = x.transpose(1, 2); // [batch, in_channels, seq_len]
        auto out = x;
        for (auto& block : blocks) {
            out = block.forward(out); // [batch, out_channels, seq_len]
        }
        out = out.transpose(1, 2); // [batch, seq_len, out_channels]
        // Take last pred_len outputs
        out = out.slice(1, -pred_len); // [batch, pred_len, out_channels]
        out = output_projection->forward(out); // [batch, pred_len, 1]
        return out;
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
            src_batch.push_back(torch::tensor(src).unsqueeze(-1)); // [seq_len, 1]
            tgt_batch.push_back(torch::tensor(tgt).unsqueeze(-1)); // [pred_len, 1]
        }
        return {torch::stack(src_batch), torch::stack(tgt_batch)}; // [batch, seq_len, 1], [batch, pred_len, 1]
    }
};

int main() {
    torch::manual_seed(0);
    std::random_device rd;
    std::mt19937 gen(rd());

    // Load dataset
    TimeSeriesDataset dataset("timeseries_data.csv", 64, 16); // seq_len=64, pred_len=16
    if (dataset.sequences.empty()) {
        std::cerr << "No valid sequences found in dataset." << std::endl;
        return -1;
    }

    // Initialize model
    TCN model(
        1, 32, 3, 4, 64, 16 // in_channels, out_channels, kernel_size, num_layers, seq_len, pred_len
    );
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    model.to(device);

    // Optimizer
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.001));

    // Training loop
    const int epochs = 30;
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
            auto output = model.forward(src); // [batch, pred_len, 1]
            auto loss = criterion->forward(output, tgt);
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model.parameters(), 1.0);
            optimizer.step();

            total_loss += loss.item<float>();
            num_batches++;
        }

        std::cout << "Epoch: " << epoch + 1 << ", Loss: " << total_loss / num_batches << std::endl;
    }

    // Save model
    torch::save(model, "tcn_model.pt");

    return 0;
}

namespace xt::models
{
    TCN::TCN(int num_classes, int in_channels)
    {
    }

    TCN::TCN(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void TCN::reset()
    {
    }

    auto TCN::forward(std::initializer_list<std::any> tensors) -> std::any
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
