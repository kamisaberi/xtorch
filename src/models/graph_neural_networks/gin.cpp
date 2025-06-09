#include "include/models/graph_neural_networks/gin.h"


using namespace std;

#include <torch/torch.h>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>

// Graph time-series data structure
struct GraphTimeSeries {
    torch::Tensor node_features; // [num_nodes, seq_len, feature_dim]
    torch::Tensor edge_index;   // [2, num_edges]
    torch::Tensor targets;      // [num_nodes, pred_len]
};

// GIN Layer
struct GINLayerImpl : torch::nn::Module {
    torch::nn::Linear linear{nullptr};
    GINLayerImpl(int in_channels, int out_channels) {
        linear = register_module("linear", torch::nn::Linear(in_channels, out_channels));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor edge_index) {
        // x: [num_nodes, seq_len, in_channels]
        auto num_nodes = x.size(0);
        auto seq_len = x.size(1);
        auto h = linear(x); // [num_nodes, seq_len, out_channels]

        // Adjacency matrix (undirected)
        torch::Tensor adj = torch::zeros({num_nodes, num_nodes}, x.options());
        auto edge_accessor = edge_index.accessor<int64_t, 2>();
        for (int64_t i = 0; i < edge_index.size(1); ++i) {
            int64_t src = edge_accessor[0][i];
            int64_t dst = edge_accessor[1][i];
            adj[src][dst] = 1.0;
            adj[dst][src] = 1.0;
        }

        // Add self-loops
        adj.diagonal() = 1.0;

        // GIN update: h_i = MLP((1 + ε) * x_i + ∑_{j∈N(i)} x_j)
        torch::Tensor neighbor_sum = torch::matmul(adj, x); // [num_nodes, seq_len, in_channels]
        h = torch::relu(h + neighbor_sum); // Simplified ε=0 for demo
        return h; // [num_nodes, seq_len, out_channels]
    }
};
TORCH_MODULE(GINLayer);

// ProbSparse Self-Attention
struct ProbSparseAttentionImpl : torch::nn::Module {
    torch::nn::Linear q_linear{nullptr}, k_linear{nullptr}, v_linear{nullptr};
    int d_model, n_heads;
    float factor;

    ProbSparseAttentionImpl(int d_model_, int n_heads_, float factor_ = 5.0)
        : d_model(d_model_), n_heads(n_heads_), factor(factor_) {
        q_linear = register_module("q_linear", torch::nn::Linear(d_model, d_model));
        k_linear = register_module("k_linear", torch::nn::Linear(d_model, d_model));
        v_linear = register_module("v_linear", torch::nn::Linear(d_model, d_model));
    }

    torch::Tensor forward(torch::Tensor x) {
        // x: [num_nodes, seq_len, d_model]
        auto batch_size = x.size(0);
        auto seq_len = x.size(1);
        auto q = q_linear(x).view({batch_size, seq_len, n_heads, d_model / n_heads}).transpose(1, 2);
        auto k = k_linear(x).view({batch_size, seq_len, n_heads, d_model / n_heads}).transpose(1, 2);
        auto v = v_linear(x).view({batch_size, seq_len, n_heads, d_model / n_heads}).transpose(1, 2);

        // ProbSparse: Sample top queries
        auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(d_model / n_heads);
        auto max_scores = scores.max(-1).values; // [batch_size, n_heads, seq_len]
        auto threshold = max_scores.mean(-1, true) - factor * max_scores.std(-1, true);
        auto mask = (scores >= threshold.unsqueeze(-1)).to(torch::kFloat32);
        scores = scores * mask - (1 - mask) * 1e9;
        auto attn = torch::softmax(scores, -1);

        auto out = torch::matmul(attn, v).transpose(1, 2).contiguous().view({batch_size, seq_len, d_model});
        return out;
    }
};
TORCH_MODULE(ProbSparseAttention);

// Informer Encoder Layer
struct InformerEncoderLayerImpl : torch::nn::Module {
    ProbSparseAttention attn{nullptr};
    torch::nn::Linear ff1{nullptr}, ff2{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};

    InformerEncoderLayerImpl(int d_model, int n_heads, int d_ff) {
        attn = register_module("attn", ProbSparseAttention(d_model, n_heads));
        ff1 = register_module("ff1", torch::nn::Linear(d_model, d_ff));
        ff2 = register_module("ff2", torch::nn::Linear(d_ff, d_model));
        norm1 = register_module("norm1", torch::nn::LayerNorm({d_model}));
        norm2 = register_module("norm2", torch::nn::LayerNorm({d_model}));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = x + attn(x);
        x = norm1(x);
        x = x + ff2(torch::relu(ff1(x)));
        x = norm2(x);
        return x;
    }
};
TORCH_MODULE(InformerEncoderLayer);

// Informer-GIN Model
struct InformerGINImpl : torch::nn::Module {
    GINLayer gin{nullptr};
    torch::nn::Linear embed{nullptr};
    std::vector<InformerEncoderLayer> encoder_layers;
    torch::nn::Linear decoder{nullptr};
    int seq_len, pred_len;

    InformerGINImpl(int in_channels, int d_model, int n_heads, int d_ff, int n_layers, int pred_len_)
        : seq_len(96), pred_len(pred_len_) {
        gin = register_module("gin", GINLayer(in_channels, d_model));
        embed = register_module("embed", torch::nn::Linear(d_model, d_model));
        for (int i = 0; i < n_layers; ++i) {
            auto layer = InformerEncoderLayer(d_model, n_heads, d_ff);
            encoder_layers.push_back(register_module("encoder_" + std::to_string(i), layer));
        }
        decoder = register_module("decoder", torch::nn::Linear(d_model, pred_len));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor edge_index) {
        // x: [num_nodes, seq_len, in_channels]
        x = gin(x, edge_index); // [num_nodes, seq_len, d_model]
        x = embed(x);
        for (auto& layer : encoder_layers) {
            x = layer(x);
        }
        x = decoder(x.mean(1)); // [num_nodes, pred_len]
        return x;
    }
};
TORCH_MODULE(InformerGIN);

// Synthetic dataset loader
std::vector<GraphTimeSeries> load_dataset(int num_graphs, int num_nodes, int seq_len, int feature_dim, int pred_len) {
    std::vector<GraphTimeSeries> dataset;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    for (int i = 0; i < num_graphs; ++i) {
        GraphTimeSeries g;
        g.node_features = torch::rand({num_nodes, seq_len, feature_dim});
        std::vector<int64_t> edges;
        for (int j = 0; j < num_nodes; ++j) {
            for (int k = j + 1; k < num_nodes; ++k) {
                if (dist(gen) > 0.5) {
                    edges.push_back(j);
                    edges.push_back(k);
                    edges.push_back(k);
                    edges.push_back(j);
                }
            }
        }
        g.edge_index = torch::tensor(edges).reshape({2, -1});
        g.targets = torch::rand({num_nodes, pred_len});
        dataset.push_back(g);
    }
    return dataset;
}

int main() {
    // Device
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Training on: " << (device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

    // Hyperparameters
    int in_channels = 16;
    int d_model = 64;
    int n_heads = 8;
    int d_ff = 256;
    int n_layers = 3;
    int seq_len = 96;
    int pred_len = 24;
    int num_epochs = 100;
    float learning_rate = 0.001;
    int batch_size = 32;

    // Model and optimizer
    auto model = InformerGIN(in_channels, d_model, n_heads, d_ff, n_layers, pred_len);
    model->to(device);
    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(learning_rate));

    // Load dataset
    auto dataset = load_dataset(1000, 20, seq_len, in_channels, pred_len);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(dataset.begin(), dataset.end(), gen);

    // Split dataset
    int train_size = 800;
    std::vector<GraphTimeSeries> train_set(dataset.begin(), dataset.begin() + train_size);
    std::vector<GraphTimeSeries> test_set(dataset.begin() + train_size, dataset.end());

    // Training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        float total_loss = 0.0;

        std::shuffle(train_set.begin(), train_set.end(), gen);
        for (size_t i = 0; i < train_set.size(); i += batch_size) {
            optimizer.zero_grad();
            std::vector<torch::Tensor> outputs, targets;

            for (size_t j = i; j < std::min(i + batch_size, train_set.size()); ++j) {
                auto graph = train_set[j];
                graph.node_features = graph.node_features.to(device);
                graph.edge_index = graph.edge_index.to(device);
                graph.targets = graph.targets.to(device);

                auto out = model->forward(graph.node_features, graph.edge_index);
                outputs.push_back(out);
                targets.push_back(graph.targets);
            }

            auto output = torch::stack(outputs);
            auto target = torch::stack(targets);
            auto loss = torch::mse_loss(output, target);
            loss.backward();
            optimizer.step();
            total_loss += loss.item<float>();
        }

        // Evaluation
        model->eval();
        float test_loss = 0.0;
        torch::NoGradGuard no_grad;
        for (const auto& graph : test_set) {
            graph.node_features = graph.node_features.to(device);
            graph.edge_index = graph.edge_index.to(device);
            graph.targets = graph.targets.to(device);

            auto out = model->forward(graph.node_features, graph.edge_index);
            test_loss += torch::mse_loss(out, graph.targets).item<float>();
        }

        std::cout << "Epoch: " << epoch + 1
                  << " | Train Loss: " << total_loss / (train_set.size() / batch_size)
                  << " | Test Loss: " << test_loss / test_set.size()
                  << std::endl;
    }

    return 0 ;
}


namespace xt::models
{
    GIN::GIN(int num_classes, int in_channels)
    {
    }

    GIN::GIN(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void GIN::reset()
    {
    }

    auto GIN::forward(std::initializer_list<std::any> tensors) -> std::any
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
