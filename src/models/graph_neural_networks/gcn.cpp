#include <models/graph_neural_networks/gcn.h>


using namespace std;


// #include <torch/torch.h>
// #include <vector>
// #include <random>
// #include <iostream>
//
// // Graph data structure
// struct Graph {
//     torch::Tensor node_features; // [num_nodes, feature_dim]
//     torch::Tensor edge_index;   // [2, num_edges]
//     torch::Tensor adj;          // [num_nodes, num_nodes] adjacency matrix
//     torch::Tensor labels;       // [num_nodes] for node classification
//     torch::Tensor train_mask;   // [num_nodes] boolean mask
//     torch::Tensor test_mask;    // [num_nodes] boolean mask
// };
//
// // GCN Layer
// struct GCNLayerImpl : torch::nn::Module {
//     torch::nn::Linear linear{nullptr};
//     GCNLayerImpl(int in_channels, int out_channels) {
//         linear = register_module("linear", torch::nn::Linear(in_channels, out_channels));
//     }
//
//     torch::Tensor forward(torch::Tensor x, torch::Tensor adj) {
//         // GCN: x' = D^-0.5 * A * D^-0.5 * X * W
//         auto h = linear(x); // [num_nodes, out_channels]
//
//         // Degree normalization
//         auto degree = adj.sum(1);
//         auto degree_inv_sqrt = torch::pow(degree + 1e-10, -0.5);
//         auto norm = degree_inv_sqrt.unsqueeze(1) * degree_inv_sqrt.unsqueeze(0);
//         auto norm_adj = norm * adj;
//
//         // Graph convolution
//         h = torch::matmul(norm_adj, h);
//         return torch::relu(h);
//     }
// };
// TORCH_MODULE(GCNLayer);
//
// // GCN Model
// struct GCNImpl : torch::nn::Module {
//     GCNLayer gcn1{nullptr}, gcn2{nullptr};
//     GCNImpl(int in_channels, int hidden_channels, int out_channels) {
//         gcn1 = register_module("gcn1", GCNLayer(in_channels, hidden_channels));
//         gcn2 = register_module("gcn2", GCNLayer(hidden_channels, out_channels));
//     }
//
//     torch::Tensor forward(torch::Tensor x, torch::Tensor adj) {
//         x = torch::relu(gcn1(x, adj));
//         x = gcn2(x, adj);
//         return x; // [num_nodes, out_channels]
//     }
// };
// TORCH_MODULE(GCN);
//
// // Synthetic dataset loader (placeholder for Cora-like dataset)
// Graph load_dataset(int num_nodes, int feature_dim, int num_classes) {
//     Graph g;
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<float> dist(0.0, 1.0);
//     std::uniform_int_distribution<int> label_dist(0, num_classes - 1);
//
//     // Node features
//     g.node_features = torch::rand({num_nodes, feature_dim});
//
//     // Random edges (undirected)
//     std::vector<int64_t> edges;
//     g.adj = torch::zeros({num_nodes, num_nodes}, torch::kFloat32);
//     for (int i = 0; i < num_nodes; ++i) {
//         for (int j = i + 1; j < num_nodes; ++j) {
//             if (dist(gen) > 0.8) {
//                 edges.push_back(i);
//                 edges.push_back(j);
//                 edges.push_back(j);
//                 edges.push_back(i);
//                 g.adj[i][j] = 1.0;
//                 g.adj[j][i] = 1.0;
//             }
//         }
//     }
//     g.edge_index = torch::tensor(edges).reshape({2, -1});
//
//     // Add self-loops
//     for (int i = 0; i < num_nodes; ++i) {
//         g.adj[i][i] = 1.0;
//     }
//
//     // Labels
//     g.labels = torch::zeros({num_nodes}, torch::kInt64);
//     for (int i = 0; i < num_nodes; ++i) {
//         g.labels[i] = label_dist(gen);
//     }
//
//     // Train/test masks
//     g.train_mask = torch::zeros({num_nodes}, torch::kBool);
//     g.test_mask = torch::zeros({num_nodes}, torch::kBool);
//     std::uniform_int_distribution<int> idx_dist(0, num_nodes - 1);
//     for (int i = 0; i < num_nodes * 0.6; ++i) {
//         g.train_mask[idx_dist(gen)] = true;
//     }
//     for (int i = 0; i < num_nodes; ++i) {
//         if (!g.train_mask[i].item<bool>()) {
//             g.test_mask[i] = true;
//         }
//     }
//
//     return g;
// }
//
// int main() {
//     // Device
//     auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
//     std::cout << "Training on: " << (device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;
//
//     // Hyperparameters
//     int in_channels = 16;
//     int hidden_channels = 16;
//     int out_channels = 7; // Number of classes (e.g., Cora has 7 classes)
//     int num_epochs = 200;
//     float learning_rate = 0.01;
//     float weight_decay = 0.0005;
//
//     // Model and optimizer
//     auto model = GCN(in_channels, hidden_channels, out_channels);
//     model->to(device);
//     auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay));
//
//     // Load dataset
//     auto graph = load_dataset(1000, in_channels, out_channels);
//     graph.node_features = graph.node_features.to(device);
//     graph.edge_index = graph.edge_index.to(device);
//     graph.adj = graph.adj.to(device);
//     graph.labels = graph.labels.to(device);
//     graph.train_mask = graph.train_mask.to(device);
//     graph.test_mask = graph.test_mask.to(device);
//
//     // Training loop
//     for (int epoch = 0; epoch < num_epochs; ++epoch) {
//         model->train();
//         optimizer.zero_grad();
//
//         auto out = model->forward(graph.node_features, graph.adj);
//         auto loss = torch::nn::functional::cross_entropy(out.index({graph.train_mask}), graph.labels.index({graph.train_mask}));
//         loss.backward();
//         optimizer.step();
//
//         // Evaluation
//         model->eval();
//         torch::NoGradGuard no_grad;
//         auto test_out = model->forward(graph.node_features, graph.adj);
//         auto test_pred = test_out.argmax(-1);
//         auto test_correct = (test_pred.index({graph.test_mask}) == graph.labels.index({graph.test_mask})).sum().item<int>();
//         auto test_total = graph.test_mask.sum().item<int>();
//         auto train_correct = (out.argmax(-1).index({graph.train_mask}) == graph.labels.index({graph.train_mask})).sum().item<int>();
//         auto train_total = graph.train_mask.sum().item<int>();
//
//         std::cout << "Epoch: " << epoch + 1
//                   << " | Train Loss: " << loss.item<float>()
//                   << " | Train Acc: " << static_cast<float>(train_correct) / train_total
//                   << " | Test Acc: " << static_cast<float>(test_correct) / test_total
//                   << std::endl;
//     }
//
//     return 0;
// }


namespace xt::models
{
    GCN::GCN(int num_classes, int in_channels)
    {
    }

    GCN::GCN(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void GCN::reset()
    {
    }

    auto GCN::forward(std::initializer_list<std::any> tensors) -> std::any
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
