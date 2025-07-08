#include "include/models/computer_vision/image_classification/dense_net.h"


using namespace std;



// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
// #include <string>
// #include <filesystem>
// #include <fstream>
// #include <sstream>
// #include <random>
//
// // Dense Layer (Bottleneck: 1x1 conv -> 3x3 conv)
// struct DenseLayerImpl : torch::nn::Module {
//     DenseLayerImpl(int in_channels, int growth_rate) {
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(in_channels));
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 4 * growth_rate, 1).bias(false)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(4 * growth_rate));
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(4 * growth_rate, growth_rate, 3).padding(1).bias(false)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto out = torch::relu(bn1->forward(x));
//         out = conv1->forward(out);
//         out = torch::relu(bn2->forward(out));
//         out = conv2->forward(out);
//         return torch::cat({x, out}, 1); // Concatenate input with output
//     }
//
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
// };
// TORCH_MODULE(DenseLayer);
//
// // Dense Block
// struct DenseBlockImpl : torch::nn::Module {
//     DenseBlockImpl(int num_layers, int in_channels, int growth_rate) {
//         for (int i = 0; i < num_layers; ++i) {
//             layers->push_back(DenseLayer(in_channels + i * growth_rate, growth_rate));
//             register_module("denselayer_" + std::to_string(i), layers->back());
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         for (auto& layer : *layers) {
//             x = layer->forward(x);
//         }
//         return x;
//     }
//
//     torch::nn::ModuleList layers{torch::nn::ModuleList()};
// };
// TORCH_MODULE(DenseBlock);
//
// // Transition Layer (1x1 conv + 2x2 avg pool)
// struct TransitionLayerImpl : torch::nn::Module {
//     TransitionLayerImpl(int in_channels, int out_channels) {
//         bn = register_module("bn", torch::nn::BatchNorm2d(in_channels));
//         conv = register_module("conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, out_channels, 1).bias(false)));
//         pool = register_module("pool", torch::nn::AvgPool2d(
//             torch::nn::AvgPool2dOptions(2).stride(2)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(bn->forward(x));
//         x = conv->forward(x);
//         x = pool->forward(x);
//         return x;
//     }
//
//     torch::nn::BatchNorm2d bn{nullptr};
//     torch::nn::Conv2d conv{nullptr};
//     torch::nn::AvgPool2d pool{nullptr};
// };
// TORCH_MODULE(TransitionLayer);
//
// // DenseNet121
// struct DenseNet121Impl : torch::nn::Module {
//     DenseNet121Impl(int num_classes = 10, int growth_rate = 32, int init_channels = 64) {
//         // Initial conv layer
//         conv0 = register_module("conv0", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, init_channels, 3).stride(1).padding(1).bias(false)));
//         bn0 = register_module("bn0", torch::nn::BatchNorm2d(init_channels));
//
//         // Dense blocks and transition layers
//         int num_features = init_channels;
//         dense1 = register_module("dense1", DenseBlock(/*num_layers*/6, num_features, growth_rate));
//         num_features += 6 * growth_rate;
//         trans1 = register_module("trans1", TransitionLayer(num_features, num_features / 2));
//         num_features /= 2;
//
//         dense2 = register_module("dense2", DenseBlock(/*num_layers*/12, num_features, growth_rate));
//         num_features += 12 * growth_rate;
//         trans2 = register_module("trans2", TransitionLayer(num_features, num_features / 2));
//         num_features /= 2;
//
//         dense3 = register_module("dense3", DenseBlock(/*num_layers*/24, num_features, growth_rate));
//         num_features += 24 * growth_rate;
//         trans3 = register_module("trans3", TransitionLayer(num_features, num_features / 2));
//         num_features /= 2;
//
//         dense4 = register_module("dense4", DenseBlock(/*num_layers*/16, num_features, growth_rate));
//         num_features += 16 * growth_rate;
//
//         // Final layers
//         bn_final = register_module("bn_final", torch::nn::BatchNorm2d(num_features));
//         fc = register_module("fc", torch::nn::Linear(num_features, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(bn0->forward(conv0->forward(x))); // [batch, 64, 32, 32]
//         x = dense1->forward(x);
//         x = trans1->forward(x); // [batch, num_features/2, 16, 16]
//         x = dense2->forward(x);
//         x = trans2->forward(x); // [batch, num_features/2, 8, 8]
//         x = dense3->forward(x);
//         x = trans3->forward(x); // [batch, num_features/2, 4, 4]
//         x = dense4->forward(x);
//         x = torch::relu(bn_final->forward(x));
//         x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, num_features, 1, 1]
//         x = x.view({x.size(0), -1}); // [batch, num_features]
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d conv0{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn_final{nullptr};
//     DenseBlock dense1{nullptr}, dense2{nullptr}, dense3{nullptr}, dense4{nullptr};
//     TransitionLayer trans1{nullptr}, trans2{nullptr}, trans3{nullptr};
//     torch::nn::Linear fc{nullptr};
// };
// TORCH_MODULE(DenseNet121);
//
// // Toy Dataset Loader (simulates CIFAR-10-like data: 32x32 RGB images)
// class ImageDataset : public torch::data::Dataset<ImageDataset> {
// public:
//     ImageDataset(const std::string& data_dir, int image_size, int num_classes)
//         : image_size_(image_size), num_classes_(num_classes) {
//         std::random_device rd;
//         rng_ = std::mt19937(rd());
//         for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
//             if (entry.path().extension() == ".txt") {
//                 image_files_.push_back(entry.path().string());
//             }
//         }
//     }
//
//     torch::data::Example<> get(size_t index) override {
//         std::ifstream file(image_files_[index]);
//         if (!file.is_open()) {
//             throw std::runtime_error("Failed to open file: " + image_files_[index]);
//         }
//         std::vector<float> image(3 * image_size_ * image_size_);
//         std::string line;
//         // Read image data (simulated: 3 channels, 32x32)
//         for (int c = 0; c < 3; ++c) {
//             std::getline(file, line);
//             std::istringstream iss(line);
//             for (int i = 0; i < image_size_ * image_size_; ++i) {
//                 iss >> image[c * image_size_ * image_size_ + i];
//             }
//         }
//         // Read label
//         std::getline(file, line);
//         int label = std::stoi(line);
//
//         auto image_tensor = torch::tensor(image).view({3, image_size_, image_size_});
//         auto label_tensor = torch::tensor(label, torch::kInt64);
//         return {image_tensor.unsqueeze(0), label_tensor.unsqueeze(0)};
//     }
//
//     torch::optional<size_t> size() const override {
//         return image_files_.size();
//     }
//
// private:
//     std::vector<std::string> image_files_;
//     int image_size_, num_classes_;
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
//         const int num_classes = 10;
//         const int batch_size = 64;
//         const int num_epochs = 10;
//         const float learning_rate = 0.001;
//         const int image_size = 32;
//         const int growth_rate = 32;
//         const int init_channels = 64;
//
//         // Initialize model
//         DenseNet121 model(num_classes, growth_rate, init_channels);
//         model->to(device);
//
//         // Optimizer
//         torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));
//
//         // Dataset and DataLoader
//         auto dataset = ImageDataset("./data/images", image_size, num_classes)
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
//                 auto images = batch.data.to(device);
//                 auto labels = batch.target.to(device).squeeze(1);
//
//                 optimizer.zero_grad();
//                 auto output = model->forward(images);
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
//                 torch::save(model, "densenet121_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: densenet121_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "densenet121_final.pt");
//         std::cout << "Saved final model: densenet121_final.pt" << std::endl;
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
//
//     return 0;
// }




// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
// #include <string>
// #include <filesystem>
// #include <fstream>
// #include <sstream>
// #include <random>
//
// // Dense Layer (Bottleneck: 1x1 conv -> 3x3 conv)
// struct DenseLayerImpl : torch::nn::Module {
//     DenseLayerImpl(int in_channels, int growth_rate) {
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(in_channels));
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 4 * growth_rate, 1).bias(false)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(4 * growth_rate));
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(4 * growth_rate, growth_rate, 3).padding(1).bias(false)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto out = torch::relu(bn1->forward(x));
//         out = conv1->forward(out);
//         out = torch::relu(bn2->forward(out));
//         out = conv2->forward(out);
//         return torch::cat({x, out}, 1); // Concatenate input with output
//     }
//
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
// };
// TORCH_MODULE(DenseLayer);
//
// // Dense Block
// struct DenseBlockImpl : torch::nn::Module {
//     DenseBlockImpl(int num_layers, int in_channels, int growth_rate) {
//         for (int i = 0; i < num_layers; ++i) {
//             layers->push_back(DenseLayer(in_channels + i * growth_rate, growth_rate));
//             register_module("denselayer_" + std::to_string(i), layers->back());
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         for (auto& layer : *layers) {
//             x = layer->forward(x);
//         }
//         return x;
//     }
//
//     torch::nn::ModuleList layers{torch::nn::ModuleList()};
// };
// TORCH_MODULE(DenseBlock);
//
// // Transition Layer (1x1 conv + 2x2 avg pool)
// struct TransitionLayerImpl : torch::nn::Module {
//     TransitionLayerImpl(int in_channels, int out_channels) {
//         bn = register_module("bn", torch::nn::BatchNorm2d(in_channels));
//         conv = register_module("conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, out_channels, 1).bias(false)));
//         pool = register_module("pool", torch::nn::AvgPool2d(
//             torch::nn::AvgPool2dOptions(2).stride(2)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(bn->forward(x));
//         x = conv->forward(x);
//         x = pool->forward(x);
//         return x;
//     }
//
//     torch::nn::BatchNorm2d bn{nullptr};
//     torch::nn::Conv2d conv{nullptr};
//     torch::nn::AvgPool2d pool{nullptr};
// };
// TORCH_MODULE(TransitionLayer);
//
// // DenseNet169
// struct DenseNet169Impl : torch::nn::Module {
//     DenseNet169Impl(int num_classes = 10, int growth_rate = 32, int init_channels = 64) {
//         // Initial conv layer
//         conv0 = register_module("conv0", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, init_channels, 3).stride(1).padding(1).bias(false)));
//         bn0 = register_module("bn0", torch::nn::BatchNorm2d(init_channels));
//
//         // Dense blocks and transition layers
//         int num_features = init_channels;
//         dense1 = register_module("dense1", DenseBlock(/*num_layers*/6, num_features, growth_rate));
//         num_features += 6 * growth_rate;
//         trans1 = register_module("trans1", TransitionLayer(num_features, num_features / 2));
//         num_features /= 2;
//
//         dense2 = register_module("dense2", DenseBlock(/*num_layers*/12, num_features, growth_rate));
//         num_features += 12 * growth_rate;
//         trans2 = register_module("trans2", TransitionLayer(num_features, num_features / 2));
//         num_features /= 2;
//
//         dense3 = register_module("dense3", DenseBlock(/*num_layers*/32, num_features, growth_rate));
//         num_features += 32 * growth_rate;
//         trans3 = register_module("trans3", TransitionLayer(num_features, num_features / 2));
//         num_features /= 2;
//
//         dense4 = register_module("dense4", DenseBlock(/*num_layers*/32, num_features, growth_rate));
//         num_features += 32 * growth_rate;
//
//         // Final layers
//         bn_final = register_module("bn_final", torch::nn::BatchNorm2d(num_features));
//         fc = register_module("fc", torch::nn::Linear(num_features, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(bn0->forward(conv0->forward(x))); // [batch, 64, 32, 32]
//         x = dense1->forward(x);
//         x = trans1->forward(x); // [batch, num_features/2, 16, 16]
//         x = dense2->forward(x);
//         x = trans2->forward(x); // [batch, num_features/2, 8, 8]
//         x = dense3->forward(x);
//         x = trans3->forward(x); // [batch, num_features/2, 4, 4]
//         x = dense4->forward(x);
//         x = torch::relu(bn_final->forward(x));
//         x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, num_features, 1, 1]
//         x = x.view({x.size(0), -1}); // [batch, num_features]
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d conv0{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn_final{nullptr};
//     DenseBlock dense1{nullptr}, dense2{nullptr}, dense3{nullptr}, dense4{nullptr};
//     TransitionLayer trans1{nullptr}, trans2{nullptr}, trans3{nullptr};
//     torch::nn::Linear fc{nullptr};
// };
// TORCH_MODULE(DenseNet169);
//
// // Toy Dataset Loader (simulates CIFAR-10-like data: 32x32 RGB images)
// class ImageDataset : public torch::data::Dataset<ImageDataset> {
// public:
//     ImageDataset(const std::string& data_dir, int image_size, int num_classes)
//         : image_size_(image_size), num_classes_(num_classes) {
//         std::random_device rd;
//         rng_ = std::mt19937(rd());
//         for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
//             if (entry.path().extension() == ".txt") {
//                 image_files_.push_back(entry.path().string());
//             }
//         }
//     }
//
//     torch::data::Example<> get(size_t index) override {
//         std::ifstream file(image_files_[index]);
//         if (!file.is_open()) {
//             throw std::runtime_error("Failed to open file: " + image_files_[index]);
//         }
//         std::vector<float> image(3 * image_size_ * image_size_);
//         std::string line;
//         // Read image data (simulated: 3 channels, 32x32)
//         for (int c = 0; c < 3; ++c) {
//             std::getline(file, line);
//             std::istringstream iss(line);
//             for (int i = 0; i < image_size_ * image_size_; ++i) {
//                 iss >> image[c * image_size_ * image_size_ + i];
//             }
//         }
//         // Read label
//         std::getline(file, line);
//         int label = std::stoi(line);
//
//         auto image_tensor = torch::tensor(image).view({3, image_size_, image_size_});
//         auto label_tensor = torch::tensor(label, torch::kInt64);
//         return {image_tensor.unsqueeze(0), label_tensor.unsqueeze(0)};
//     }
//
//     torch::optional<size_t> size() const override {
//         return image_files_.size();
//     }
//
// private:
//     std::vector<std::string> image_files_;
//     int image_size_, num_classes_;
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
//         const int num_classes = 10;
//         const int batch_size = 64;
//         const int num_epochs = 10;
//         const float learning_rate = 0.001;
//         const int image_size = 32;
//         const int growth_rate = 32;
//         const int init_channels = 64;
//
//         // Initialize model
//         DenseNet169 model(num_classes, growth_rate, init_channels);
//         model->to(device);
//
//         // Optimizer
//         torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));
//
//         // Dataset and DataLoader
//         auto dataset = ImageDataset("./data/images", image_size, num_classes)
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
//                 auto images = batch.data.to(device);
//                 auto labels = batch.target.to(device).squeeze(1);
//
//                 optimizer.zero_grad();
//                 auto output = model->forward(images);
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
//                 torch::save(model, "densenet169_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: densenet169_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "densenet169_final.pt");
//         std::cout << "Saved final model: densenet169_final.pt" << std::endl;
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
//
//     return 0;
// }







// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
// #include <string>
// #include <filesystem>
// #include <fstream>
// #include <sstream>
// #include <random>
//
// // Dense Layer (Bottleneck: 1x1 conv -> 3x3 conv)
// struct DenseLayerImpl : torch::nn::Module {
//     DenseLayerImpl(int in_channels, int growth_rate) {
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(in_channels));
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 4 * growth_rate, 1).bias(false)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(4 * growth_rate));
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(4 * growth_rate, growth_rate, 3).padding(1).bias(false)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto out = torch::relu(bn1->forward(x));
//         out = conv1->forward(out);
//         out = torch::relu(bn2->forward(out));
//         out = conv2->forward(out);
//         return torch::cat({x, out}, 1); // Concatenate input with output
//     }
//
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
// };
// TORCH_MODULE(DenseLayer);
//
// // Dense Block
// struct DenseBlockImpl : torch::nn::Module {
//     DenseBlockImpl(int num_layers, int in_channels, int growth_rate) {
//         for (int i = 0; i < num_layers; ++i) {
//             layers->push_back(DenseLayer(in_channels + i * growth_rate, growth_rate));
//             register_module("denselayer_" + std::to_string(i), layers->back());
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         for (auto& layer : *layers) {
//             x = layer->forward(x);
//         }
//         return x;
//     }
//
//     torch::nn::ModuleList layers{torch::nn::ModuleList()};
// };
// TORCH_MODULE(DenseBlock);
//
// // Transition Layer (1x1 conv + 2x2 avg pool)
// struct TransitionLayerImpl : torch::nn::Module {
//     TransitionLayerImpl(int in_channels, int out_channels) {
//         bn = register_module("bn", torch::nn::BatchNorm2d(in_channels));
//         conv = register_module("conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, out_channels, 1).bias(false)));
//         pool = register_module("pool", torch::nn::AvgPool2d(
//             torch::nn::AvgPool2dOptions(2).stride(2)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(bn->forward(x));
//         x = conv->forward(x);
//         x = pool->forward(x);
//         return x;
//     }
//
//     torch::nn::BatchNorm2d bn{nullptr};
//     torch::nn::Conv2d conv{nullptr};
//     torch::nn::AvgPool2d pool{nullptr};
// };
// TORCH_MODULE(TransitionLayer);
//
// // DenseNet201
// struct DenseNet201Impl : torch::nn::Module {
//     DenseNet201Impl(int num_classes = 10, int growth_rate = 32, int init_channels = 64) {
//         // Initial conv layer
//         conv0 = register_module("conv0", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, init_channels, 3).stride(1).padding(1).bias(false)));
//         bn0 = register_module("bn0", torch::nn::BatchNorm2d(init_channels));
//
//         // Dense blocks and transition layers
//         int num_features = init_channels;
//         dense1 = register_module("dense1", DenseBlock(/*num_layers*/6, num_features, growth_rate));
//         num_features += 6 * growth_rate;
//         trans1 = register_module("trans1", TransitionLayer(num_features, num_features / 2));
//         num_features /= 2;
//
//         dense2 = register_module("dense2", DenseBlock(/*num_layers*/12, num_features, growth_rate));
//         num_features += 12 * growth_rate;
//         trans2 = register_module("trans2", TransitionLayer(num_features, num_features / 2));
//         num_features /= 2;
//
//         dense3 = register_module("dense3", DenseBlock(/*num_layers*/48, num_features, growth_rate));
//         num_features += 48 * growth_rate;
//         trans3 = register_module("trans3", TransitionLayer(num_features, num_features / 2));
//         num_features /= 2;
//
//         dense4 = register_module("dense4", DenseBlock(/*num_layers*/32, num_features, growth_rate));
//         num_features += 32 * growth_rate;
//
//         // Final layers
//         bn_final = register_module("bn_final", torch::nn::BatchNorm2d(num_features));
//         fc = register_module("fc", torch::nn::Linear(num_features, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(bn0->forward(conv0->forward(x))); // [batch, 64, 32, 32]
//         x = dense1->forward(x);
//         x = trans1->forward(x); // [batch, num_features/2, 16, 16]
//         x = dense2->forward(x);
//         x = trans2->forward(x); // [batch, num_features/2, 8, 8]
//         x = dense3->forward(x);
//         x = trans3->forward(x); // [batch, num_features/2, 4, 4]
//         x = dense4->forward(x);
//         x = torch::relu(bn_final->forward(x));
//         x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, num_features, 1, 1]
//         x = x.view({x.size(0), -1}); // [batch, num_features]
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d conv0{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn_final{nullptr};
//     DenseBlock dense1{nullptr}, dense2{nullptr}, dense3{nullptr}, dense4{nullptr};
//     TransitionLayer trans1{nullptr}, trans2{nullptr}, trans3{nullptr};
//     torch::nn::Linear fc{nullptr};
// };
// TORCH_MODULE(DenseNet201);
//
// // Toy Dataset Loader (simulates CIFAR-10-like data: 32x32 RGB images)
// class ImageDataset : public torch::data::Dataset<ImageDataset> {
// public:
//     ImageDataset(const std::string& data_dir, int image_size, int num_classes)
//         : image_size_(image_size), num_classes_(num_classes) {
//         std::random_device rd;
//         rng_ = std::mt19937(rd());
//         for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
//             if (entry.path().extension() == ".txt") {
//                 image_files_.push_back(entry.path().string());
//             }
//         }
//     }
//
//     torch::data::Example<> get(size_t index) override {
//         std::ifstream file(image_files_[index]);
//         if (!file.is_open()) {
//             throw std::runtime_error("Failed to open file: " + image_files_[index]);
//         }
//         std::vector<float> image(3 * image_size_ * image_size_);
//         std::string line;
//         // Read image data (simulated: 3 channels, 32x32)
//         for (int c = 0; c < 3; ++c) {
//             std::getline(file, line);
//             std::istringstream iss(line);
//             for (int i = 0; i < image_size_ * image_size_; ++i) {
//                 iss >> image[c * image_size_ * image_size_ + i];
//             }
//         }
//         // Read label
//         std::getline(file, line);
//         int label = std::stoi(line);
//
//         auto image_tensor = torch::tensor(image).view({3, image_size_, image_size_});
//         auto label_tensor = torch::tensor(label, torch::kInt64);
//         return {image_tensor.unsqueeze(0), label_tensor.unsqueeze(0)};
//     }
//
//     torch::optional<size_t> size() const override {
//         return image_files_.size();
//     }
//
// private:
//     std::vector<std::string> image_files_;
//     int image_size_, num_classes_;
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
//         const int num_classes = 10;
//         const int batch_size = 64;
//         const int num_epochs = 10;
//         const float learning_rate = 0.001;
//         const int image_size = 32;
//         const int growth_rate = 32;
//         const int init_channels = 64;
//
//         // Initialize model
//         DenseNet201 model(num_classes, growth_rate, init_channels);
//         model->to(device);
//
//         // Optimizer
//         torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));
//
//         // Dataset and DataLoader
//         auto dataset = ImageDataset("./data/images", image_size, num_classes)
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
//                 auto images = batch.data.to(device);
//                 auto labels = batch.target.to(device).squeeze(1);
//
//                 optimizer.zero_grad();
//                 auto output = model->forward(images);
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
//                 torch::save(model, "densenet201_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: densenet201_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "densenet201_final.pt");
//         std::cout << "Saved final model: densenet201_final.pt" << std::endl;
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
//
//     return 0;
// }







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
// // Dense Layer (Bottleneck: 1x1 conv -> 3x3 conv)
// struct DenseLayerImpl : torch::nn::Module {
//     DenseLayerImpl(int in_channels, int growth_rate) {
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(in_channels));
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 4 * growth_rate, 1).bias(false)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(4 * growth_rate));
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(4 * growth_rate, growth_rate, 3).padding(1).bias(false)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto out = torch::relu(bn1->forward(x));
//         out = conv1->forward(out);
//         out = torch::relu(bn2->forward(out));
//         out = conv2->forward(out);
//         return torch::cat({x, out}, 1); // Concatenate input with output
//     }
//
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
// };
// TORCH_MODULE(DenseLayer);
//
// // Dense Block
// struct DenseBlockImpl : torch::nn::Module {
//     DenseBlockImpl(int num_layers, int in_channels, int growth_rate) {
//         for (int i = 0; i < num_layers; ++i) {
//             layers->push_back(DenseLayer(in_channels + i * growth_rate, growth_rate));
//             register_module("denselayer_" + std::to_string(i), layers->back());
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         for (auto& layer : *layers) {
//             x = layer->forward(x);
//         }
//         return x;
//     }
//
//     torch::nn::ModuleList layers{torch::nn::ModuleList()};
// };
// TORCH_MODULE(DenseBlock);
//
// // Transition Layer (1x1 conv + 2x2 avg pool)
// struct TransitionLayerImpl : torch::nn::Module {
//     TransitionLayerImpl(int in_channels, int out_channels) {
//         bn = register_module("bn", torch::nn::BatchNorm2d(in_channels));
//         conv = register_module("conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, out_channels, 1).bias(false)));
//         pool = register_module("pool", torch::nn::AvgPool2d(
//             torch::nn::AvgPool2dOptions(2).stride(2)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(bn->forward(x));
//         x = conv->forward(x);
//         x = pool->forward(x);
//         return x;
//     }
//
//     torch::nn::BatchNorm2d bn{nullptr};
//     torch::nn::Conv2d conv{nullptr};
//     torch::nn::AvgPool2d pool{nullptr};
// };
// TORCH_MODULE(TransitionLayer);
//
// // DenseNet264
// struct DenseNet264Impl : torch::nn::Module {
//     DenseNet264Impl(int num_classes = 10, int growth_rate = 32, int init_channels = 64) {
//         // Initial conv layer
//         conv0 = register_module("conv0", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, init_channels, 3).stride(1).padding(1).bias(false)));
//         bn0 = register_module("bn0", torch::nn::BatchNorm2d(init_channels));
//
//         // Dense blocks and transition layers
//         int num_features = init_channels;
//         dense1 = register_module("dense1", DenseBlock(/*num_layers*/6, num_features, growth_rate));
//         num_features += 6 * growth_rate;
//         trans1 = register_module("trans1", TransitionLayer(num_features, num_features / 2));
//         num_features /= 2;
//
//         dense2 = register_module("dense2", DenseBlock(/*num_layers*/12, num_features, growth_rate));
//         num_features += 12 * growth_rate;
//         trans2 = register_module("trans2", TransitionLayer(num_features, num_features / 2));
//         num_features /= 2;
//
//         dense3 = register_module("dense3", DenseBlock(/*num_layers*/64, num_features, growth_rate));
//         num_features += 64 * growth_rate;
//         trans3 = register_module("trans3", TransitionLayer(num_features, num_features / 2));
//         num_features /= 2;
//
//         dense4 = register_module("dense4", DenseBlock(/*num_layers*/32, num_features, growth_rate));
//         num_features += 32 * growth_rate;
//
//         // Final layers
//         bn_final = register_module("bn_final", torch::nn::BatchNorm2d(num_features));
//         fc = register_module("fc", torch::nn::Linear(num_features, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(bn0->forward(conv0->forward(x))); // [batch, 64, 32, 32]
//         x = dense1->forward(x);
//         x = trans1->forward(x); // [batch, num_features/2, 16, 16]
//         x = dense2->forward(x);
//         x = trans2->forward(x); // [batch, num_features/2, 8, 8]
//         x = dense3->forward(x);
//         x = trans3->forward(x); // [batch, num_features/2, 4, 4]
//         x = dense4->forward(x);
//         x = torch::relu(bn_final->forward(x));
//         x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, num_features, 1, 1]
//         x = x.view({x.size(0), -1}); // [batch, num_features]
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d conv0{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn_final{nullptr};
//     DenseBlock dense1{nullptr}, dense2{nullptr}, dense3{nullptr}, dense4{nullptr};
//     TransitionLayer trans1{nullptr}, trans2{nullptr}, trans3{nullptr};
//     torch::nn::Linear fc{nullptr};
// };
// TORCH_MODULE(DenseNet264);
//
// // Toy Dataset Loader (simulates CIFAR-10-like data: 32x32 RGB images)
// class ImageDataset : public torch::data::Dataset<ImageDataset> {
// public:
//     ImageDataset(const std::string& data_dir, int image_size, int num_classes)
//         : image_size_(image_size), num_classes_(num_classes) {
//         std::random_device rd;
//         rng_ = std::mt19937(rd());
//         for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
//             if (entry.path().extension() == ".txt") {
//                 image_files_.push_back(entry.path().string());
//             }
//         }
//     }
//
//     torch::data::Example<> get(size_t index) override {
//         std::ifstream file(image_files_[index]);
//         if (!file.is_open()) {
//             throw std::runtime_error("Failed to open file: " + image_files_[index]);
//         }
//         std::vector<float> image(3 * image_size_ * image_size_);
//         std::string line;
//         // Read image data (simulated: 3 channels, 32x32)
//         for (int c = 0; c < 3; ++c) {
//             std::getline(file, line);
//             std::istringstream iss(line);
//             for (int i = 0; i < image_size_ * image_size_; ++i) {
//                 iss >> image[c * image_size_ * image_size_ + i];
//             }
//         }
//         // Read label
//         std::getline(file, line);
//         int label = std::stoi(line);
//
//         auto image_tensor = torch::tensor(image).view({3, image_size_, image_size_});
//         auto label_tensor = torch::tensor(label, torch::kInt64);
//         return {image_tensor.unsqueeze(0), label_tensor.unsqueeze(0)};
//     }
//
//     torch::optional<size_t> size() const override {
//         return image_files_.size();
//     }
//
// private:
//     std::vector<std::string> image_files_;
//     int image_size_, num_classes_;
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
//         const int num_classes = 10;
//         const int batch_size = 64;
//         const int num_epochs = 10;
//         const float learning_rate = 0.001;
//         const int image_size = 32;
//         const int growth_rate = 32;
//         const int init_channels = 64;
//
//         // Initialize model
//         DenseNet264 model(num_classes, growth_rate, init_channels);
//         model->to(device);
//
//         // Optimizer
//         torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));
//
//         // Dataset and DataLoader
//         auto dataset = ImageDataset("./data/images", image_size, num_classes)
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
//                 auto images = batch.data.to(device);
//                 auto labels = batch.target.to(device).squeeze(1);
//
//                 optimizer.zero_grad();
//                 auto output = model->forward(images);
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
//                 torch::save(model, "densenet264_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: densenet264_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "densenet264_final.pt");
//         std::cout << "Saved final model: densenet264_final.pt" << std::endl;
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
//
//     return 0;
// }



namespace xt::models {

    // Dense Layer (Bottleneck: 1x1 conv -> 3x3 conv)
    DenseLayerImpl::DenseLayerImpl(int in_channels, int growth_rate) {
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(in_channels));
        conv1 = register_module("conv1", torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, 4 * growth_rate, 1).bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(4 * growth_rate));
        conv2 = register_module("conv2", torch::nn::Conv2d(
                torch::nn::Conv2dOptions(4 * growth_rate, growth_rate, 3).padding(1).bias(false)));
    }


    torch::Tensor DenseLayerImpl::forward(torch::Tensor x) {
        auto out = torch::relu(bn1->forward(x));
        out = conv1->forward(out);
        out = torch::relu(bn2->forward(out));
        out = conv2->forward(out);
        return torch::cat({x, out}, 1); // Concatenate input with output
    }

    // Dense Block
    DenseBlockImpl::DenseBlockImpl(int num_layers, int in_channels, int growth_rate) {
        for (int i = 0; i < num_layers; ++i) {
            layers->push_back(DenseLayer(in_channels + i * growth_rate, growth_rate));
            register_module("denselayer_" + std::to_string(i), layers->back());
        }
    }

    torch::Tensor DenseBlockImpl::forward(torch::Tensor x) {
        for (auto &layer: *layers) {
            x = layer->forward(x);
        }
        return x;
    }

    TransitionLayerImpl::TransitionLayerImpl(int in_channels, int out_channels) {
        bn = register_module("bn", torch::nn::BatchNorm2d(in_channels));
        conv = register_module("conv", torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, out_channels, 1).bias(false)));
        pool = register_module("pool", torch::nn::AvgPool2d(
                torch::nn::AvgPool2dOptions(2).stride(2)));
    }

    torch::Tensor TransitionLayerImpl::forward(torch::Tensor x) {
        x = torch::relu(bn->forward(x));
        x = conv->forward(x);
        x = pool->forward(x);
        return x;
    }

    // DenseNet121
    DenseNet121Impl::DenseNet121Impl(int num_classes = 10, int growth_rate = 32, int init_channels = 64) {
        // Initial conv layer
        conv0 = register_module("conv0", torch::nn::Conv2d(
                torch::nn::Conv2dOptions(3, init_channels, 3).stride(1).padding(1).bias(false)));
        bn0 = register_module("bn0", torch::nn::BatchNorm2d(init_channels));

        // Dense blocks and transition layers
        int num_features = init_channels;
        dense1 = register_module("dense1", DenseBlock(/*num_layers*/6, num_features, growth_rate));
        num_features += 6 * growth_rate;
        trans1 = register_module("trans1", TransitionLayer(num_features, num_features / 2));
        num_features /= 2;

        dense2 = register_module("dense2", DenseBlock(/*num_layers*/12, num_features, growth_rate));
        num_features += 12 * growth_rate;
        trans2 = register_module("trans2", TransitionLayer(num_features, num_features / 2));
        num_features /= 2;

        dense3 = register_module("dense3", DenseBlock(/*num_layers*/24, num_features, growth_rate));
        num_features += 24 * growth_rate;
        trans3 = register_module("trans3", TransitionLayer(num_features, num_features / 2));
        num_features /= 2;

        dense4 = register_module("dense4", DenseBlock(/*num_layers*/16, num_features, growth_rate));
        num_features += 16 * growth_rate;

        // Final layers
        bn_final = register_module("bn_final", torch::nn::BatchNorm2d(num_features));
        fc = register_module("fc", torch::nn::Linear(num_features, num_classes));
    }

    torch::Tensor DenseNet121Impl::forward(torch::Tensor x) {
        x = torch::relu(bn0->forward(conv0->forward(x))); // [batch, 64, 32, 32]
        x = dense1->forward(x);
        x = trans1->forward(x); // [batch, num_features/2, 16, 16]
        x = dense2->forward(x);
        x = trans2->forward(x); // [batch, num_features/2, 8, 8]
        x = dense3->forward(x);
        x = trans3->forward(x); // [batch, num_features/2, 4, 4]
        x = dense4->forward(x);
        x = torch::relu(bn_final->forward(x));
        x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, num_features, 1, 1]
        x = x.view({x.size(0), -1}); // [batch, num_features]
        x = fc->forward(x); // [batch, num_classes]
        return x;
    }


    DenseNet121::DenseNet121(int num_classes, int in_channels) {
    }

    DenseNet121::DenseNet121(int num_classes, int in_channels, std::vector <int64_t> input_shape) {
    }

    void DenseNet121::reset() {
    }

    auto DenseNet121::forward(std::initializer_list <std::any> tensors) -> std::any {
        std::vector <std::any> any_vec(tensors);

        std::vector <torch::Tensor> tensor_vec;
        for (const auto &item: any_vec) {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];

        return x;
    }

    // DenseNet169
    DenseNet169Impl::DenseNet169Impl(int num_classes = 10, int growth_rate = 32, int init_channels = 64) {
        // Initial conv layer
        conv0 = register_module("conv0", torch::nn::Conv2d(
                torch::nn::Conv2dOptions(3, init_channels, 3).stride(1).padding(1).bias(false)));
        bn0 = register_module("bn0", torch::nn::BatchNorm2d(init_channels));

        // Dense blocks and transition layers
        int num_features = init_channels;
        dense1 = register_module("dense1", DenseBlock(/*num_layers*/6, num_features, growth_rate));
        num_features += 6 * growth_rate;
        trans1 = register_module("trans1", TransitionLayer(num_features, num_features / 2));
        num_features /= 2;

        dense2 = register_module("dense2", DenseBlock(/*num_layers*/12, num_features, growth_rate));
        num_features += 12 * growth_rate;
        trans2 = register_module("trans2", TransitionLayer(num_features, num_features / 2));
        num_features /= 2;

        dense3 = register_module("dense3", DenseBlock(/*num_layers*/32, num_features, growth_rate));
        num_features += 32 * growth_rate;
        trans3 = register_module("trans3", TransitionLayer(num_features, num_features / 2));
        num_features /= 2;

        dense4 = register_module("dense4", DenseBlock(/*num_layers*/32, num_features, growth_rate));
        num_features += 32 * growth_rate;

        // Final layers
        bn_final = register_module("bn_final", torch::nn::BatchNorm2d(num_features));
        fc = register_module("fc", torch::nn::Linear(num_features, num_classes));
    }

    torch::Tensor DenseNet169Impl::forward(torch::Tensor x) {
        x = torch::relu(bn0->forward(conv0->forward(x))); // [batch, 64, 32, 32]
        x = dense1->forward(x);
        x = trans1->forward(x); // [batch, num_features/2, 16, 16]
        x = dense2->forward(x);
        x = trans2->forward(x); // [batch, num_features/2, 8, 8]
        x = dense3->forward(x);
        x = trans3->forward(x); // [batch, num_features/2, 4, 4]
        x = dense4->forward(x);
        x = torch::relu(bn_final->forward(x));
        x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, num_features, 1, 1]
        x = x.view({x.size(0), -1}); // [batch, num_features]
        x = fc->forward(x); // [batch, num_classes]
        return x;
    }

    DenseNet169::DenseNet169(int num_classes, int in_channels) {
    }

    DenseNet169::DenseNet169(int num_classes, int in_channels, std::vector <int64_t> input_shape) {
    }

    void DenseNet169::reset() {
    }

    auto DenseNet169::forward(std::initializer_list <std::any> tensors) -> std::any {
        std::vector <std::any> any_vec(tensors);

        std::vector <torch::Tensor> tensor_vec;
        for (const auto &item: any_vec) {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];

        return x;
    }


    // DenseNet201
    DenseNet201Impl::DenseNet201Impl(int num_classes = 10, int growth_rate = 32, int init_channels = 64) {
        // Initial conv layer
        conv0 = register_module("conv0", torch::nn::Conv2d(
                torch::nn::Conv2dOptions(3, init_channels, 3).stride(1).padding(1).bias(false)));
        bn0 = register_module("bn0", torch::nn::BatchNorm2d(init_channels));

        // Dense blocks and transition layers
        int num_features = init_channels;
        dense1 = register_module("dense1", DenseBlock(/*num_layers*/6, num_features, growth_rate));
        num_features += 6 * growth_rate;
        trans1 = register_module("trans1", TransitionLayer(num_features, num_features / 2));
        num_features /= 2;

        dense2 = register_module("dense2", DenseBlock(/*num_layers*/12, num_features, growth_rate));
        num_features += 12 * growth_rate;
        trans2 = register_module("trans2", TransitionLayer(num_features, num_features / 2));
        num_features /= 2;

        dense3 = register_module("dense3", DenseBlock(/*num_layers*/48, num_features, growth_rate));
        num_features += 48 * growth_rate;
        trans3 = register_module("trans3", TransitionLayer(num_features, num_features / 2));
        num_features /= 2;

        dense4 = register_module("dense4", DenseBlock(/*num_layers*/32, num_features, growth_rate));
        num_features += 32 * growth_rate;

        // Final layers
        bn_final = register_module("bn_final", torch::nn::BatchNorm2d(num_features));
        fc = register_module("fc", torch::nn::Linear(num_features, num_classes));
    }

    torch::Tensor DenseNet201Impl::forward(torch::Tensor x) {
        x = torch::relu(bn0->forward(conv0->forward(x))); // [batch, 64, 32, 32]
        x = dense1->forward(x);
        x = trans1->forward(x); // [batch, num_features/2, 16, 16]
        x = dense2->forward(x);
        x = trans2->forward(x); // [batch, num_features/2, 8, 8]
        x = dense3->forward(x);
        x = trans3->forward(x); // [batch, num_features/2, 4, 4]
        x = dense4->forward(x);
        x = torch::relu(bn_final->forward(x));
        x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, num_features, 1, 1]
        x = x.view({x.size(0), -1}); // [batch, num_features]
        x = fc->forward(x); // [batch, num_classes]
        return x;
    }


    DenseNet201::DenseNet201(int num_classes, int in_channels) {
    }

    DenseNet201::DenseNet201(int num_classes, int in_channels, std::vector <int64_t> input_shape) {
    }

    void DenseNet201::reset() {
    }

    auto DenseNet201::forward(std::initializer_list <std::any> tensors) -> std::any {
        std::vector <std::any> any_vec(tensors);

        std::vector <torch::Tensor> tensor_vec;
        for (const auto &item: any_vec) {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];

        return x;
    }


    // DenseNet264
    DenseNet264Impl::DenseNet264Impl(int num_classes = 10, int growth_rate = 32, int init_channels = 64) {
        // Initial conv layer
        conv0 = register_module("conv0", torch::nn::Conv2d(
                torch::nn::Conv2dOptions(3, init_channels, 3).stride(1).padding(1).bias(false)));
        bn0 = register_module("bn0", torch::nn::BatchNorm2d(init_channels));

        // Dense blocks and transition layers
        int num_features = init_channels;
        dense1 = register_module("dense1", DenseBlock(/*num_layers*/6, num_features, growth_rate));
        num_features += 6 * growth_rate;
        trans1 = register_module("trans1", TransitionLayer(num_features, num_features / 2));
        num_features /= 2;

        dense2 = register_module("dense2", DenseBlock(/*num_layers*/12, num_features, growth_rate));
        num_features += 12 * growth_rate;
        trans2 = register_module("trans2", TransitionLayer(num_features, num_features / 2));
        num_features /= 2;

        dense3 = register_module("dense3", DenseBlock(/*num_layers*/64, num_features, growth_rate));
        num_features += 64 * growth_rate;
        trans3 = register_module("trans3", TransitionLayer(num_features, num_features / 2));
        num_features /= 2;

        dense4 = register_module("dense4", DenseBlock(/*num_layers*/32, num_features, growth_rate));
        num_features += 32 * growth_rate;

        // Final layers
        bn_final = register_module("bn_final", torch::nn::BatchNorm2d(num_features));
        fc = register_module("fc", torch::nn::Linear(num_features, num_classes));
    }

    torch::Tensor DenseNet264Impl::forward(torch::Tensor x) {
        x = torch::relu(bn0->forward(conv0->forward(x))); // [batch, 64, 32, 32]
        x = dense1->forward(x);
        x = trans1->forward(x); // [batch, num_features/2, 16, 16]
        x = dense2->forward(x);
        x = trans2->forward(x); // [batch, num_features/2, 8, 8]
        x = dense3->forward(x);
        x = trans3->forward(x); // [batch, num_features/2, 4, 4]
        x = dense4->forward(x);
        x = torch::relu(bn_final->forward(x));
        x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, num_features, 1, 1]
        x = x.view({x.size(0), -1}); // [batch, num_features]
        x = fc->forward(x); // [batch, num_classes]
        return x;
    }


    DenseNet264::DenseNet264(int num_classes, int in_channels) {
    }

    DenseNet264::DenseNet264(int num_classes, int in_channels, std::vector <int64_t> input_shape) {
    }

    void DenseNet264::reset() {
    }

    auto DenseNet264::forward(std::initializer_list <std::any> tensors) -> std::any {
        std::vector <std::any> any_vec(tensors);

        std::vector <torch::Tensor> tensor_vec;
        for (const auto &item: any_vec) {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];

        return x;
    }


}
