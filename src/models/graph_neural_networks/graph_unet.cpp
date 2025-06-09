#include "include/models/graph_neural_networks/graph_unet.h"


using namespace std;


#include <torch/torch.h>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>

// Graph data structure
struct Graph {
    torch::Tensor node_features; // [num_nodes, feature_dim]
    torch::Tensor edge_index;   // [2, num_edges]
    torch::Tensor labels;       // [num_nodes] for node classification
    torch::Tensor train_mask;   // [num_nodes] boolean mask
    torch::Tensor test_mask;    // [num_nodes] boolean mask
};

// GCN Layer
struct GCNLayerImpl : torch::nn::Module {
    torch::nn::Linear linear{nullptr};
    GCNLayerImpl(int in_channels, int out_channels) {
        linear = register_module("linear", torch::nn::Linear(in_channels, out_channels));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor edge_index) {
        // Simplified GCN: x' = A * X * W
        auto num_nodes = x.size(0);
        auto h = linear(x); // [num_nodes, out_channels]

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

        // Degree normalization
        auto degree = adj.sum(1);
        auto degree_inv_sqrt = torch::pow(degree + 1e-10, -0.5);
        auto norm = degree_inv_sqrt.unsqueeze(1) * degree_inv_sqrt.unsqueeze(0);
        adj = norm * adj;

        // Graph convolution
        h = torch::matmul(adj, h);
        return torch::relu(h);
    }
};
TORCH_MODULE(GCNLayer);

// Top-K Pooling
struct TopKPoolingImpl : torch::nn::Module {
    float ratio;
    TopKPoolingImpl(float ratio_) : ratio(ratio_) {}

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor edge_index) {
        // x: [num_nodes, channels]
        auto num_nodes = x.size(0);
        int64_t k = static_cast<int64_t>(num_nodes * ratio);

        // Compute scores (use norm of features as score)
        auto scores = x.norm(2, 1); // [num_nodes]
        auto [top_scores, indices] = scores.topk(k); // [k]

        // Select top-k nodes
        auto x_pool = x.index_select(0, indices); // [k, channels]

        // Filter edges
        std::vector<int64_t> new_edges;
        auto edge_accessor = edge_index.accessor<int64_t, 2>();
        std::vector<int64_t> node_map(num_nodes, -1);
        for (int64_t i = 0; i < k; ++i) {
            node_map[indices[i].item<int64_t>()] = i;
        }
        for (int64_t i = 0; i < edge_index.size(1); ++i) {
            int64_t src = edge_accessor[0][i];
            int64_t dst = edge_accessor[1][i];
            if (node_map[src] != -1 && node_map[dst] != -1) {
                new_edges.push_back(node_map[src]);
                new_edges.push_back(node_map[dst]);
            }
        }
        auto edge_index_pool = new_edges.empty() ? torch::empty({2, 0}, edge_index.options())
                                                 : torch::tensor(new_edges).reshape({2, -1}).to(edge_index.device());

        return std::make_tuple(x_pool, edge_index_pool, indices);
    }
};
TORCH_MODULE(TopKPooling);

// Unpooling
struct UnpoolingImpl : torch::nn::Module {
    UnpoolingImpl() {}

    torch::Tensor forward(torch::Tensor x_pool, torch::Tensor indices, int64_t orig_num_nodes) {
        // x_pool: [k, channels], indices: [k]
        torch::Tensor x = torch::zeros({orig_num_nodes, x_pool.size(1)}, x_pool.options());
        x.index_copy_(0, indices, x_pool);
        return x;
    }
};
TORCH_MODULE(Unpooling);

// GraphUNet Model
struct GraphUNetImpl : torch::nn::Module {
    GCNLayer gcn1{nullptr}, gcn2{nullptr}, gcn3{nullptr}, gcn4{nullptr}, gcn5{nullptr};
    TopKPooling pool1{nullptr}, pool2{nullptr};
    Unpooling unpool1{nullptr}, unpool2{nullptr};
    torch::nn::Linear fc{nullptr};

    GraphUNetImpl(int in_channels, int hidden_channels, int out_channels) {
        gcn1 = register_module("gcn1", GCNLayer(in_channels, hidden_channels));
        gcn2 = register_module("gcn2", GCNLayer(hidden_channels, hidden_channels));
        gcn3 = register_module("gcn3", GCNLayer(hidden_channels, hidden_channels));
        gcn4 = register_module("gcn4", GCNLayer(hidden_channels * 2, hidden_channels));
        gcn5 = register_module("gcn5", GCNLayer(hidden_channels * 2, hidden_channels));
        pool1 = register_module("pool1", TopKPooling(0.5));
        pool2 = register_module("pool2", TopKPooling(0.5));
        unpool1 = register_module("unpool1", Unpooling());
        unpool2 = register_module("unpool2", Unpooling());
        fc = register_module("fc", torch::nn::Linear(hidden_channels, out_channels));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor edge_index) {
        auto num_nodes = x.size(0);

        // Encoder
        auto x1 = gcn1(x, edge_index); // [num_nodes, hidden_channels]
        torch::Tensor x_pool1, edge_index1, idx1;
        std::tie(x_pool1, edge_index1, idx1) = pool1(x1, edge_index);
        auto x2 = gcn2(x_pool1, edge_index1);

        torch::Tensor x_pool2, edge_index2, idx2;
        std::tie(x_pool2, edge_index2, idx2) = pool2(x2, edge_index1);
        auto x3 = gcn3(x_pool2, edge_index2);

        // Decoder
        auto x_up1 = unpool1(x3, idx2, x2.size(0));
        auto x4 = gcn4(torch::cat({x_up1, x2}, -1), edge_index1);

        auto x_up2 = unpool2(x4, idx1, num_nodes);
        auto x5 = gcn5(torch::cat({x_up2, x1}, -1), edge_index);

        // Output
        return fc(x5); // [num_nodes, out_channels]
    }
};
TORCH_MODULE(GraphUNet);

// Synthetic dataset loader (placeholder for Cora-like dataset)
Graph load_dataset(int num_nodes, int feature_dim, int num_classes) {
    Graph g;
    std::random_device rd;
    std::mt19937 gen(rd());
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
    int hidden_channels = 32;
    int out_channels = 7; // Number of classes (e.g., Cora has 7 classes)
    int num_epochs = 200;
    float learning_rate = 0.01;
    float weight_decay = 0.0005;

    // Model and optimizer
    auto model = GraphUNet(in_channels, hidden_channels, out_channels);
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
    GraphUNet::GraphUNet(int num_classes, int in_channels)
    {
    }

    GraphUNet::GraphUNet(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void GraphUNet::reset()
    {
    }

    auto GraphUNet::forward(std::initializer_list<std::any> tensors) -> std::any
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
