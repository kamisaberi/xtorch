#include <models/graph_neural_networks/gat.h>


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
//     torch::Tensor labels;       // [num_nodes] for node classification
//     torch::Tensor train_mask;   // [num_nodes] boolean mask
//     torch::Tensor test_mask;    // [num_nodes] boolean mask
// };
//
// // GAT Layer
// struct GATLayerImpl : torch::nn::Module {
//     torch::nn::Linear linear{nullptr};
//     torch::Tensor a; // Attention parameters [2 * out_channels]
//     int in_channels, out_channels;
//
//     GATLayerImpl(int in_channels_, int out_channels_, int heads = 1)
//         : in_channels(in_channels_), out_channels(out_channels_) {
//         linear = register_module("linear", torch::nn::Linear(in_channels, out_channels * heads));
//         a = register_parameter("a", torch::randn({2 * out_channels * heads}));
//     }
//
//     torch::Tensor forward(torch::Tensor x, torch::Tensor edge_index) {
//         auto num_nodes = x.size(0);
//         auto h = linear(x).view({num_nodes, -1, out_channels}); // [num_nodes, heads, out_channels]
//
//         // Compute attention coefficients
//         auto edge_accessor = edge_index.accessor<int64_t, 2>();
//         auto num_edges = edge_index.size(1);
//         torch::Tensor alpha = torch::zeros({num_edges, h.size(1)}, x.options());
//
//         for (int64_t i = 0; i < num_edges; ++i) {
//             int64_t src = edge_accessor[0][i];
//             int64_t dst = edge_accessor[1][i];
//             auto h_src = h[src]; // [heads, out_channels]
//             auto h_dst = h[dst]; // [heads, out_channels]
//             auto concat = torch::cat({h_src, h_dst}, -1); // [heads, 2 * out_channels]
//             alpha[i] = (concat * a).sum(-1); // [heads]
//         }
//
//         // Apply leaky ReLU and softmax
//         alpha = torch::leaky_relu(alpha, 0.2);
//         torch::Tensor exp_alpha = torch::exp(alpha);
//         torch::Tensor sum_alpha = torch::zeros({num_nodes, h.size(1)}, x.options());
//
//         for (int64_t i = 0; i < num_edges; ++i) {
//             int64_t dst = edge_accessor[1][i];
//             sum_alpha[dst] += exp_alpha[i];
//         }
//
//         torch::Tensor norm_alpha = torch::zeros_like(alpha);
//         for (int64_t i = 0; i < num_edges; ++i) {
//             int64_t dst = edge_accessor[1][i];
//             norm_alpha[i] = exp_alpha[i] / (sum_alpha[dst] + 1e-10);
//         }
//
//         // Message passing
//         torch::Tensor out = torch::zeros_like(h);
//         for (int64_t i = 0; i < num_edges; ++i) {
//             int64_t src = edge_accessor[0][i];
//             int64_t dst = edge_accessor[1][i];
//             out[dst] += norm_alpha[i].unsqueeze(-1) * h[src];
//         }
//
//         return out.mean(1); // Average over heads [num_nodes, out_channels]
//     }
// };
// TORCH_MODULE(GATLayer);
//
// // GAT Model
// struct GATImpl : torch::nn::Module {
//     GATLayer gat1{nullptr}, gat2{nullptr};
//     GATImpl(int in_channels, int hidden_channels, int out_channels, int heads = 8) {
//         gat1 = register_module("gat1", GATLayer(in_channels, hidden_channels, heads));
//         gat2 = register_module("gat2", GATLayer(hidden_channels, out_channels, 1));
//     }
//
//     torch::Tensor forward(torch::Tensor x, torch::Tensor edge_index) {
//         x = torch::relu(gat1(x, edge_index));
//         x = gat2(x, edge_index);
//         return x; // [num_nodes, out_channels]
//     }
// };
// TORCH_MODULE(GAT);
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
//     for (int i = 0; i < num_nodes; ++i) {
//         for (int j = i + 1; j < num_nodes; ++j) {
//             if (dist(gen) > 0.8) {
//                 edges.push_back(i);
//                 edges.push_back(j);
//                 edges.push_back(j);
//                 edges.push_back(i);
//             }
//         }
//     }
//     g.edge_index = torch::tensor(edges).reshape({2, -1});
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
//     std::cout << "Training on: " << (device == torch::kCUDA ? "CPU" : "CUDA") << std::endl;
//
//     // Hyperparameters
//     int in_channels = 16;
//     int hidden_channels = 8;
//     int out_channels = 7; // Number of classes (e.g., Cora has 7 classes)
//     int heads = 8;
//     int num_epochs = 200;
//     float learning_rate = 0.005;
//     float weight_decay = 0.0005;
//
//     // Model and optimizer
//     auto model = GAT(in_channels, hidden_channels, out_channels, heads);
//     model->to(device);
//     auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay));
//
//     // Load dataset
//     auto graph = load_dataset(1000, in_channels, out_channels);
//     graph.node_features = graph.node_features.to(device);
//     graph.edge_index = graph.edge_index.to(device);
//     graph.labels = graph.labels.to(device);
//     graph.train_mask = graph.train_mask.to(device);
//     graph.test_mask = graph.test_mask.to(device);
//
//     // Training loop
//     for (int epoch = 0; epoch < num_epochs; ++epoch) {
//         model->train();
//         optimizer.zero_grad();
//
//         auto out = model->forward(graph.node_features, graph.edge_index);
//         auto loss = torch::nn::functional::cross_entropy(out.index({graph.train_mask}), graph.labels.index({graph.train_mask}));
//         loss.backward();
//         optimizer.step();
//
//         // Evaluation
//         model->eval();
//         torch::NoGradGuard no_grad;
//         auto test_out = model->forward(graph.node_features, graph.edge_index);
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
    GAT::GAT(int num_classes, int in_channels)
    {
    }

    GAT::GAT(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void GAT::reset()
    {
    }

    auto GAT::forward(std::initializer_list<std::any> tensors) -> std::any
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
