#include "include/models/graph_neural_networks/diff_pool.h"


using namespace std;


#include <torch/torch.h>
#include <vector>
#include <string>
#include <random>
#include <iostream>

// Graph data structure
struct Graph {
    torch::Tensor node_features; // [num_nodes, feature_dim]
    torch::Tensor edge_index;   // [2, num_edges]
    torch::Tensor y;            // Graph label (scalar for classification)
};

// GCN Layer
struct GCNImpl : torch::nn::Module {
    torch::nn::Linear linear{nullptr};
    GCNImpl(int in_channels, int out_channels) {
        linear = register_module("linear", torch::nn::Linear(in_channels, out_channels));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor edge_index) {
        // Simplified GCN: x' = A * X * W
        // Assume edge_index is [2, num_edges], x is [num_nodes, in_channels]
        auto num_nodes = x.size(0);
        auto h = linear(x); // [num_nodes, out_channels]

        // Adjacency matrix (simplified, assuming undirected graph)
        torch::Tensor adj = torch::zeros({num_nodes, num_nodes}, x.options());
        auto edge_accessor = edge_index.accessor<int64_t, 2>();
        for (int64_t i = 0; i < edge_index.size(1); ++i) {
            int64_t src = edge_accessor[0][i];
            int64_t dst = edge_accessor[1][i];
            adj[src][dst] = 1.0;
            adj[dst][src] = 1.0; // Undirected
        }

        // Add self-loops
        adj.diagonal() = 1.0;

        // Degree normalization
        torch::Tensor degree = adj.sum(1);
        torch::Tensor degree_inv_sqrt = torch::pow(degree, -0.5);
        torch::Tensor norm = degree_inv_sqrt.unsqueeze(1) * degree_inv_sqrt.unsqueeze(0);

        // Normalized adjacency: D^-0.5 * A * D^-0.5
        adj = norm * adj;

        // Graph convolution
        h = torch::matmul(adj, h);
        return torch::relu(h);
    }
};
TORCH_MODULE(GCN);

// DiffPool Layer
struct DiffPoolImpl : torch::nn::Module {
    GCN gcn{nullptr};
    torch::nn::Linear embed{nullptr};
    DiffPoolImpl(int in_channels, int out_channels, int assign_channels) {
        gcn = register_module("gcn", GCN(in_channels, out_channels));
        embed = register_module("embed", torch::nn::Linear(out_channels, assign_channels));
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor edge_index) {
        // x: [num_nodes, in_channels]
        // edge_index: [2, num_edges]
        auto z = gcn(x, edge_index); // [num_nodes, out_channels]
        auto s = torch::softmax(embed(z), 0); // [num_nodes, assign_channels]

        // Compute coarsened adjacency
        auto num_nodes = x.size(0);
        auto assign_channels = s.size(1);
        torch::Tensor new_adj = torch::zeros({assign_channels, assign_channels}, x.options());
        auto edge_accessor = edge_index.accessor<int64_t, 2>();
        for (int64_t i = 0; i < edge_index.size(1); ++i) {
            int64_t src = edge_accessor[0][i];
            int64_t dst = edge_accessor[1][i];
            auto s_src = s[src]; // [assign_channels]
            auto s_dst = s[dst]; // [assign_channels]
            new_adj += torch::ger(s_src, s_dst); // Outer product
        }

        // Coarsened node features
        auto x_new = torch::matmul(s.t(), z); // [assign_channels, out_channels]

        // Compute pooling loss (entropy regularization)
        auto entropy_loss = -torch::sum(s * torch::log(s + 1e-10)) / num_nodes;

        return std::make_tuple(x_new, new_adj, entropy_loss);
    }
};
TORCH_MODULE(DiffPool);

// GNN Model with DiffPool
struct GNNImpl : torch::nn::Module {
    GCN gcn1{nullptr}, gcn2{nullptr};
    DiffPool diffpool{nullptr};
    torch::nn::Linear fc{nullptr};
    GNNImpl(int in_channels, int hidden_channels, int num_clusters) {
        gcn1 = register_module("gcn1", GCN(in_channels, hidden_channels));
        diffpool = register_module("diffpool", DiffPool(hidden_channels, hidden_channels, num_clusters));
        gcn2 = register_module("gcn2", GCN(hidden_channels, hidden_channels));
        fc = register_module("fc", torch::nn::Linear(hidden_channels, 2)); // Binary classification
    }

    torch::Tensor forward(const Graph& graph, float& pool_loss) {
        auto x = graph.node_features;
        auto edge_index = graph.edge_index;

        x = gcn1(x, edge_index);
        torch::Tensor x_new, new_adj, entropy_loss;
        std::tie(x_new, new_adj, entropy_loss) = diffpool(x, edge_index);
        pool_loss = entropy_loss.item<float>();

        // Compute new edge_index from new_adj
        auto edge_list = torch::nonzero(new_adj > 0.1).t(); // Threshold for sparsity
        x_new = gcn2(x_new, edge_list);
        x_new = x_new.mean(0); // Global mean pooling
        return fc(x_new);
    }
};
TORCH_MODULE(GNN);

// Synthetic dataset loader (placeholder for TUDataset)
std::vector<Graph> load_dataset(int num_graphs, int num_nodes, int feature_dim) {
    std::vector<Graph> dataset;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    std::uniform_int_distribution<int> label_dist(0, 1);

    for (int i = 0; i < num_graphs; ++i) {
        Graph g;
        g.node_features = torch::rand({num_nodes, feature_dim});
        // Random edge_index
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
        g.y = torch::tensor({label_dist(gen)}, torch::kInt64);
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
    int hidden_channels = 32;
    int num_clusters = 10;
    int num_epochs = 100;
    float learning_rate = 0.001;
    int batch_size = 32;

    // Model and optimizer
    auto model = GNN(in_channels, hidden_channels, num_clusters);
    model->to(device);
    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(learning_rate));

    // Load dataset
    auto dataset = load_dataset(1000, 20, in_channels); // 1000 graphs, 20 nodes each
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(dataset.begin(), dataset.end(), gen);

    // Split dataset
    int train_size = 800;
    std::vector<Graph> train_set(dataset.begin(), dataset.begin() + train_size);
    std::vector<Graph> test_set(dataset.begin() + train_size, dataset.end());

    // Training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        float total_loss = 0.0;
        int correct = 0, total = 0;

        std::shuffle(train_set.begin(), train_set.end(), gen);
        for (size_t i = 0; i < train_set.size(); i += batch_size) {
            optimizer.zero_grad();
            float pool_loss = 0.0;
            std::vector<torch::Tensor> outputs;

            for (size_t j = i; j < std::min(i + batch_size, train_set.size()); ++j) {
                auto graph = train_set[j];
                graph.node_features = graph.node_features.to(device);
                graph.edge_index = graph.edge_index.to(device);
                graph.y = graph.y.to(device);

                auto out = model->forward(graph, pool_loss);
                outputs.push_back(out);
                total++;
                correct += (out.argmax(-1) == graph.y).item<int>();
            }

            auto output = torch::stack(outputs);
            auto labels = torch::cat(
                std::vector<torch::Tensor>(
                    train_set.begin() + i,
                    train_set.begin() + std::min(i + batch_size, train_set.size())
                ) | std::views::transform([](const Graph& g) { return g.y; })
            ).to(device);

            auto loss = torch::nn::functional::cross_entropy(output, labels) + pool_loss;
            loss.backward();
            optimizer.step();
            total_loss += loss.item<float>();
        }

        // Evaluation
        model->eval();
        int test_correct = 0, test_total = 0;
        float test_loss = 0.0;
        for (const auto& graph : test_set) {
            graph.node_features = graph.node_features.to(device);
            graph.edge_index = graph.edge_index.to(device);
            graph.y = graph.y.to(device);
            float pool_loss = 0.0;

            torch::NoGradGuard no_grad;
            auto out = model->forward(graph, pool_loss);
            test_loss += torch::nn::functional::cross_entropy(out.unsqueeze(0), graph.y).item<float>();
            test_correct += (out.argmax(-1) == graph.y).item<int>();
            test_total++;
        }

        std::cout << "Epoch: " << epoch + 1
                  << " | Train Loss: " << total_loss / (train_set.size() / batch_size)
                  << " | Train Acc: " << static_cast<float>(correct) / total
                  << " | Test Loss: " << test_loss / test_total
                  << " | Test Acc: " << static_cast<float>(test_correct) / test_total
                  << std::endl;
    }

    return 0;
}


namespace xt::models
{
    DiffPool::DiffPool(int num_classes, int in_channels)
    {
    }

    DiffPool::DiffPool(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DiffPool::reset()
    {
    }

    auto DiffPool::forward(std::initializer_list<std::any> tensors) -> std::any
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
