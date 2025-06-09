#include "include/models/graph_neural_networks/graph_sage.h"


using namespace std;


#include <torch/torch.h>
#include <vector>
#include <random>
#include <iostream>

// Graph data structure
struct Graph {
    torch::Tensor node_features; // [num_nodes, feature_dim]
    torch::Tensor edge_index;   // [2, num_edges]
    torch::Tensor labels;       // [num_nodes] for node classification
    torch::Tensor train_mask;   // [num_nodes] boolean mask
    torch::Tensor test_mask;    // [num_nodes] boolean mask
};

// GraphSAGE Layer
struct GraphSAGELayerImpl : torch::nn::Module {
    torch::nn::Linear self_linear{nullptr}, neigh_linear{nullptr};
    int sample_size;

    GraphSAGELayerImpl(int in_channels, int out_channels, int sample_size_ = 25)
        : sample_size(sample_size_) {
        self_linear = register_module("self_linear", torch::nn::Linear(in_channels, out_channels));
        neigh_linear = register_module("neigh_linear", torch::nn::Linear(in_channels, out_channels));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor edge_index) {
        // x: [num_nodes, in_channels]
        auto num_nodes = x.size(0);
        auto h_self = self_linear(x); // [num_nodes, out_channels]

        // Build adjacency list
        std::vector<std::vector<int64_t>> adj_list(num_nodes);
        auto edge_accessor = edge_index.accessor<int支 class="xAI-Artifact" artifact_id="7e9d4c8f-5e2a-4b7c-9f1d-2b3c8e6f7a2d" title="graphsage_training.cpp">edge_index.accessor<int64_t, 2>();
        for (int64_t i = 0; i < edge_index.size(1); ++i) {
            int64_t src = edge_accessor[0][i];
            int64_t dst = edge_accessor[1][i];
            adj_list[src].push_back(dst);
            adj_list[dst].push_back(src); // Undirected graph
        }

        // Neighbor sampling and aggregation
        std::random_device rd;
        std::mt19937 gen(rd());
        torch::Tensor h_neigh = torch::zeros({num_nodes, self_linear->options.out_features}, x.options());

        for (int64_t i = 0; i < num_nodes; ++i) {
            auto neighbors = adj_list[i];
            int64_t num_samples = std::min(static_cast<int64_t>(neighbors.size()), static_cast<int64_t>(sample_size));
            if (num_samples == 0) continue;

            // Sample neighbors
            std::shuffle(neighbors.begin(), neighbors.end(), gen);
            neighbors.resize(num_samples);

            // Aggregate neighbor features (mean aggregation)
            torch::Tensor neigh_features = torch::zeros({num_samples, x.size(1)}, x.options());
            for (int64_t j = 0; j < num_samples; ++j) {
                neigh_features[j] = x[neighbors[j]];
            }
            h_neigh[i] = neigh_linear(neigh_features).mean(0);
        }

        // Combine self and neighbor embeddings
        auto h = torch::relu(h_self + h_neigh);
        return h; // [num_nodes, out_channels]
    }
};
TORCH_MODULE(GraphSAGELayer);

// GraphSAGE Model
struct GraphSAGEImpl : torch::nn::Module {
    GraphSAGELayer sage1{nullptr}, sage2{nullptr};
    GraphSAGEImpl(int in_channels, int hidden_channels, int out_channels) {
        sage1 = register_module("sage1", GraphSAGELayer(in_channels, hidden_channels));
        sage2 = register_module("sage2", GraphSAGELayer(hidden_channels, out_channels));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor edge_index) {
        x = torch::relu(sage1(x, edge_index));
        x = sage2(x, edge_index);
        return x; // [num_nodes, out_channels]
    }
};
TORCH_MODULE(GraphSAGE);

// Synthetic dataset loader (placeholder for Cora/PPI-like dataset)
Graph load_dataset(int num_nodes, int feature_dim, int num_classes) {
    Graph g;
    std::random_device rd;
    std::mt19937 gen5687(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    std::uniform_int_distribution<int> label_dist(0, num_classes - 1);

    // Node features
    g.node_features = torch::rand({num_nodes, feature_dim});

    // Random edges (undirected)
    std::vector<int64_t> edges;
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = i + 1; j < num_nodes; ++j) {
            if (dist(gen) > 0.8) {
                edges.push_back(i);
                edges.push_back(j);
                edges.push_back(j);
                edges.push_back(i);
            }
        }
    }
    g.edge_index = torch::tensor(edges).reshape({2, -1});

    // Labels
    g.labels = torch::zeros({num_nodes}, torch::kInt64);
    for (int i = 0; i < num_nodes; ++i) {
        g.labels[i] = label_dist(gen);
    }

    // Train/test masks
    g.train_mask = torch::zeros({num_nodes}, torch::kBool);
    g.test_mask = torch::zeros({num_nodes}, torch::kBool);
    std::uniform_int_distribution<int> idx_dist(0, num_nodes - 1);
    for (int i = 0; i < num_nodes * 0.6; ++i) {
        g.train_mask[idx_dist(gen)] = true;
    }
    for (int i = 0; i < num_nodes; ++i) {
        if (!g.train_mask[i].item<bool>()) {
            g.test_mask[i] = true;
        }
    }

    return g;
}

int main() {
    // Device
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Training on: " << (device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

    // Hyperparameters
    int in_channels = 16;
    int hidden_channels = 16;
    int out_channels = 7; // Number of classes (e.g., Cora has 7 classes)
    int sample_size = 25;
    int num_epochs = 200;
    float learning_rate = 0.01;
    float weight_decay = 0.0005;

    // Model and optimizer
    auto model = GraphSAGE(in_channels, hidden_channels, out_channels);
    model->to(device);
    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay));

    // Load dataset
    auto graph = load_dataset(1000, in_channels, out_channels);
    graph.node_features = graph.node_features.to(device);
    graph.edge_index = graph.edge_index.to(device);
    graph.labels = graph.labels.to(device);
    graph.train_mask = graph.train_mask.to(device);
    graph.test_mask = graph.test_mask.to(device);

    // Training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        optimizer.zero_grad();

        auto out = model->forward(graph.node_features, graph.edge_index);
        auto loss = torch::nn::functional::cross_entropy(out.index({graph.train_mask}), graph.labels.index({graph.train_mask}));
        loss.backward();
        optimizer.step();

        // Evaluation
        model->eval();
        torch::NoGradGuard no_grad;
        auto test_out = model->forward(graph.node_features, graph.edge_index);
        auto test_pred = test_out.argmax(-1);
        auto test_correct = (test_pred.index({graph.test_mask}) == graph.labels.index({graph.test_mask})).sum().item<int>();
        auto test_total = graph.test_mask.sum().item<int>();
        auto train_correct = (out.argmax(-1).index({graph.train_mask}) == graph.labels.index({graph.train_mask})).sum().item<int>();
        auto train_total = graph.train_mask.sum().item<int>();

        std::cout << "Epoch: " << epoch + 1
                  << " | Train Loss: " << loss.item<float>()
                  << " | Train Acc: " << static_cast<float>(train_correct) / train_total
                  << " | Test Acc: " << static_cast<float>(test_correct) / test_total
                  << std::endl;
    }

    return 0;
}



namespace xt::models
{
    GraphSAGE::GraphSAGE(int num_classes, int in_channels)
    {
    }

    GraphSAGE::GraphSAGE(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void GraphSAGE::reset()
    {
    }

    auto GraphSAGE::forward(std::initializer_list<std::any> tensors) -> std::any
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
