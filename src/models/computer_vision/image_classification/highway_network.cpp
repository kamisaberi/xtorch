#include "include/models/computer_vision/image_classification/highway_network.h"


using namespace std;

//
// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
// #include <string>
// #include <filesystem>
// #include <fstream>
// #include <sstream>
// #include <random>
//
// // Highway Layer
// struct HighwayLayerImpl : torch::nn::Module {
//     HighwayLayerImpl(int input_size) {
//         // Transform gate: W_T * x + b_T
//         transform = register_module("transform", torch::nn::Linear(input_size, input_size));
//         // Plain layer: W * x + b
//         plain = register_module("plain", torch::nn::Linear(input_size, input_size));
//         // Initialize transform gate bias to encourage carrying input initially
//         transform->bias.data().fill_(-2.0);
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Transform gate output: sigmoid(W_T * x + b_T)
//         auto T = torch::sigmoid(transform->forward(x));
//         // Plain output: relu(W * x + b)
//         auto H = torch::relu(plain->forward(x));
//         // Highway output: T * H + (1 - T) * x
//         return T * H + (1.0 - T) * x;
//     }
//
//     torch::nn::Linear transform{nullptr}, plain{nullptr};
// };
// TORCH_MODULE(HighwayLayer);
//
// // Highway Network
// struct HighwayNetworkImpl : torch::nn::Module {
//     HighwayNetworkImpl(int input_size, int num_classes, int num_layers) {
//         // Input layer
//         input_layer = register_module("input_layer", torch::nn::Linear(input_size, input_size));
//
//         // Highway layers
//         for (int i = 0; i < num_layers; ++i) {
//             layers->push_back(HighwayLayer(input_size));
//             register_module("highway_" + std::to_string(i), layers->back());
//         }
//
//         // Output layer
//         output_layer = register_module("output_layer", torch::nn::Linear(input_size, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(input_layer->forward(x));
//         for (auto& layer : *layers) {
//             x = layer->forward(x);
//         }
//         x = output_layer->forward(x);
//         return x;
//     }
//
//     torch::nn::Linear input_layer{nullptr}, output_layer{nullptr};
//     torch::nn::ModuleList layers{torch::nn::ModuleList()};
// };
// TORCH_MODULE(HighwayNetwork);
//
// // Toy Dataset Loader (simulates MNIST-like data: 784 features, 10 classes)
// class ToyDataset : public torch::data::Dataset<ToyDataset> {
// public:
//     ToyDataset(const std::string& data_dir, int input_size, int num_classes)
//         : input_size_(input_size), num_classes_(num_classes) {
//         std::random_device rd;
//         rng_ = std::mt19937(rd());
//         for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
//             if (entry.path().extension() == ".txt") {
//                 data_files_.push_back(entry.path().string());
//             }
//         }
//     }
//
//     torch::data::Example<> get(size_t index) override {
//         std::ifstream file(data_files_[index]);
//         if (!file.is_open()) {
//             throw std::runtime_error("Failed to open file: " + data_files_[index]);
//         }
//         std::vector<float> features(input_size_);
//         std::string line;
//         std::getline(file, line);
//         std::istringstream iss(line);
//         for (int i = 0; i < input_size_; ++i) {
//             iss >> features[i];
//         }
//         std::getline(file, line);
//         int label = std::stoi(line);
//
//         auto feature_tensor = torch::tensor(features).view({1, input_size_});
//         auto label_tensor = torch::tensor(label, torch::kInt64);
//         return {feature_tensor, label_tensor.unsqueeze(0)};
//     }
//
//     torch::optional<size_t> size() const override {
//         return data_files_.size();
//     }
//
// private:
//     std::vector<std::string> data_files_;
//     int input_size_, num_classes_;
//     std::mt19937 rng_;
// };
//
// int main() {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Hyperparameters
//         const int input_size = 784; // e.g., MNIST flattened 28x28
//         const int num_classes = 10;
//         const int num_layers = 10; // Number of highway layers
//         const int batch_size = 64;
//         const int num_epochs = 10;
//         const float learning_rate = 0.001;
//
//         // Initialize model
//         HighwayNetwork model(input_size, num_classes, num_layers);
//         model->to(device);
//
//         // Optimizer
//         torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));
//
//         // Dataset and DataLoader
//         auto dataset = ToyDataset("./data/features", input_size, num_classes)
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
//             int correct = 0, total = 0;
//
//             for (auto& batch : *data_loader) {
//                 auto features = batch.data.to(device).squeeze(1); // [batch, 784]
//                 auto labels = batch.target.to(device).squeeze(1); // [batch]
//
//                 optimizer.zero_grad();
//                 auto output = model->forward(features);
//                 auto loss = torch::nn::functional::cross_entropy(output, labels);
//
//                 loss.backward();
//                 torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
//                 optimizer.step();
//
//                 total_loss += loss.item<float>();
//                 batch_count++;
//
//                 auto predicted = output.argmax(1);
//                 total += labels.size(0);
//                 correct += predicted.eq(labels).sum().item<int64_t>();
//             }
//
//             float avg_loss = total_loss / batch_count;
//             float accuracy = 100.0 * correct / total;
//             std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs
//                       << "] Loss: " << avg_loss
//                       << ", Accuracy: " << accuracy << "%" << std::endl;
//
//             // Save checkpoint every 5 epochs
//             if ((epoch + 1) % 5 == 0) {
//                 torch::save(model, "highway_network_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: highway_network_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "highway_network_final.pt");
//         std::cout << "Saved final model: highway_network_final.pt" << std::endl;
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
//
//     return 0;
// }


namespace xt::models {
    // Highway Layer
    HighwayLayerImpl::HighwayLayerImpl(int input_size) {
        // Transform gate: W_T * x + b_T
        transform = register_module("transform", torch::nn::Linear(input_size, input_size));
        // Plain layer: W * x + b
        plain = register_module("plain", torch::nn::Linear(input_size, input_size));
        // Initialize transform gate bias to encourage carrying input initially
        transform->bias.data().fill_(-2.0);
    }

    torch::Tensor HighwayLayerImpl::forward(torch::Tensor x) {
        // Transform gate output: sigmoid(W_T * x + b_T)
        auto T = torch::sigmoid(transform->forward(x));
        // Plain output: relu(W * x + b)
        auto H = torch::relu(plain->forward(x));
        // Highway output: T * H + (1 - T) * x
        return T * H + (1.0 - T) * x;
    }

    HighwayNetworkImpl::HighwayNetworkImpl(int input_size, int num_classes, int num_layers) {
        // Input layer
        input_layer = register_module("input_layer", torch::nn::Linear(input_size, input_size));

        // Highway layers
        for (int i = 0; i < num_layers; ++i) {
            layers->push_back(HighwayLayer(input_size));
            register_module("highway_" + std::to_string(i), layers[layers->size()-1]);
        }

        // Output layer
        output_layer = register_module("output_layer", torch::nn::Linear(input_size, num_classes));
    }

    torch::Tensor HighwayNetworkImpl::forward(torch::Tensor x) {
        x = torch::relu(input_layer->forward(x));
        for (auto &layer: *layers) {
            x = layer->forward(x);
        }
        x = output_layer->forward(x);
        return x;
    }

    // HighwayNetwork::HighwayNetwork(int num_classes, int in_channels) {
    // }
    //
    // HighwayNetwork::HighwayNetwork(int num_classes, int in_channels, std::vector <int64_t> input_shape) {
    // }
    //
    // void HighwayNetwork::reset() {
    // }
    //
    // auto HighwayNetwork::forward(std::initializer_list <std::any> tensors) -> std::any {
    //     std::vector <std::any> any_vec(tensors);
    //
    //     std::vector <torch::Tensor> tensor_vec;
    //     for (const auto &item: any_vec) {
    //         tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
    //     }
    //
    //     torch::Tensor x = tensor_vec[0];
    //
    //     return x;
    // }
}
