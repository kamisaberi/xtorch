#include "include/models/computer_vision/image_classification/efficient_net.h"


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
// // Swish activation (x * sigmoid(x))
// torch::Tensor swish(torch::Tensor x) {
//     return x * torch::sigmoid(x);
// }
//
// // Squeeze-and-Excitation Block
// struct SEBlockImpl : torch::nn::Module {
//     SEBlockImpl(int in_channels, int reduction) {
//         fc1 = register_module("fc1", torch::nn::Linear(in_channels, in_channels / reduction));
//         fc2 = register_module("fc2", torch::nn::Linear(in_channels / reduction, in_channels));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto batch = x.size(0);
//         auto channels = x.size(1);
//         auto avg = torch::avg_pool2d(x, {x.size(2), x.size(3)}).view({batch, channels});
//         auto out = torch::relu(fc1->forward(avg));
//         out = torch::sigmoid(fc2->forward(out)).view({batch, channels, 1, 1});
//         return x * out;
//     }
//
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr};
// };
// TORCH_MODULE(SEBlock);
//
// // MBConv Block (Inverted Residual with Depthwise Separable Conv)
// struct MBConvBlockImpl : torch::nn::Module {
//     MBConvBlockImpl(int in_channels, int out_channels, int expansion, int kernel_size, int stride, int reduction) {
//         int expanded_channels = in_channels * expansion;
//         bool has_se = reduction > 0;
//
//         if (expansion != 1) {
//             expand_conv = register_module("expand_conv", torch::nn::Conv2d(
//                 torch::nn::Conv2dOptions(in_channels, expanded_channels, 1).bias(false)));
//             bn0 = register_module("bn0", torch::nn::BatchNorm2d(expanded_channels));
//         }
//
//         depthwise_conv = register_module("depthwise_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(expanded_channels, expanded_channels, kernel_size)
//                 .stride(stride).padding(kernel_size / 2).groups(expanded_channels).bias(false)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(expanded_channels));
//
//         if (has_se) {
//             se = register_module("se", SEBlock(expanded_channels, reduction));
//         }
//
//         pointwise_conv = register_module("pointwise_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(expanded_channels, out_channels, 1).bias(false)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
//
//         skip_connection = (in_channels == out_channels && stride == 1);
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto out = x;
//         if (expand_conv) {
//             out = swish(bn0->forward(expand_conv->forward(out)));
//         }
//         out = swish(bn1->forward(depthwise_conv->forward(out)));
//         if (se) {
//             out = se->forward(out);
//         }
//         out = bn2->forward(pointwise_conv->forward(out));
//         if (skip_connection) {
//             out += x; // Residual connection
//         }
//         return out;
//     }
//
//     torch::nn::Conv2d expand_conv{nullptr}, depthwise_conv{nullptr}, pointwise_conv{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr}, bn2{nullptr};
//     SEBlock se{nullptr};
//     bool skip_connection;
// };
// TORCH_MODULE(MBConvBlock);
//
// // EfficientNetB0
// struct EfficientNetB0Impl : torch::nn::Module {
//     EfficientNetB0Impl(int num_classes = 10) {
//         // Initial stem
//         stem_conv = register_module("stem_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, 32, 3).stride(1).padding(1).bias(false))); // Simplified stride
//         bn0 = register_module("bn0", torch::nn::BatchNorm2d(32));
//
//         // MBConv blocks configuration: {num_repeats, in_channels, out_channels, expansion, kernel_size, stride, se_reduction}
//         std::vector<std::tuple<int, int, int, int, int, int, int>> config = {
//             {1, 32, 16, 1, 3, 1, 4},   // Stage 1
//             {2, 16, 24, 6, 3, 2, 4},   // Stage 2
//             {2, 24, 40, 6, 5, 2, 4},   // Stage 3
//             {3, 40, 80, 6, 3, 2, 4},   // Stage 4
//             {3, 80, 112, 6, 5, 1, 4},  // Stage 5
//             {4, 112, 192, 6, 5, 2, 4}, // Stage 6
//             {1, 192, 320, 6, 3, 1, 4}  // Stage 7
//         };
//
//         int stage_idx = 0;
//         for (const auto& [num_repeats, in_ch, out_ch, expansion, kernel, stride, reduction] : config) {
//             for (int i = 0; i < num_repeats; ++i) {
//                 int s = (i == 0) ? stride : 1;
//                 blocks->push_back(MBConvBlock(in_ch, out_ch, expansion, kernel, s, reduction));
//                 register_module("block_" + std::to_string(stage_idx) + "_" + std::to_string(i), blocks->back());
//                 in_ch = out_ch;
//             }
//             stage_idx++;
//         }
//
//         // Head
//         head_conv = register_module("head_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(320, 1280, 1).bias(false)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(1280));
//         fc = register_module("fc", torch::nn::Linear(1280, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = swish(bn0->forward(stem_conv->forward(x))); // [batch, 32, 32, 32]
//         for (auto& block : *blocks) {
//             x = block->forward(x);
//         }
//         x = swish(bn1->forward(head_conv->forward(x)));
//         x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, 1280, 1, 1]
//         x = x.view({x.size(0), -1}); // [batch, 1280]
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
//     torch::nn::Linear fc{nullptr};
//     torch::nn::ModuleList blocks{torch::nn::ModuleList()};
// };
// TORCH_MODULE(EfficientNetB0);
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
//         for (int c = 0; c < 3; ++c) {
//             std::getline(file, line);
//             std::istringstream iss(line);
//             for (int i = 0; i < image_size_ * image_size_; ++i) {
//                 iss >> image[c * image_size_ * image_size_ + i];
//             }
//         }
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
//
//         // Initialize model
//         EfficientNetB0 model(num_classes);
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
//                 torch::save(model, "efficientnetb0_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: efficientnetb0_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "efficientnetb0_final.pt");
//         std::cout << "Saved final model: efficientnetb0_final.pt" << std::endl;
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
// // Swish activation (x * sigmoid(x))
// torch::Tensor swish(torch::Tensor x) {
//     return x * torch::sigmoid(x);
// }
//
// // Squeeze-and-Excitation Block
// struct SEBlockImpl : torch::nn::Module {
//     SEBlockImpl(int in_channels, int reduction) {
//         fc1 = register_module("fc1", torch::nn::Linear(in_channels, in_channels / reduction));
//         fc2 = register_module("fc2", torch::nn::Linear(in_channels / reduction, in_channels));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto batch = x.size(0);
//         auto channels = x.size(1);
//         auto avg = torch::avg_pool2d(x, {x.size(2), x.size(3)}).view({batch, channels});
//         auto out = torch::relu(fc1->forward(avg));
//         out = torch::sigmoid(fc2->forward(out)).view({batch, channels, 1, 1});
//         return x * out;
//     }
//
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr};
// };
// TORCH_MODULE(SEBlock);
//
// // MBConv Block (Inverted Residual with Depthwise Separable Conv)
// struct MBConvBlockImpl : torch::nn::Module {
//     MBConvBlockImpl(int in_channels, int out_channels, int expansion, int kernel_size, int stride, int reduction) {
//         int expanded_channels = in_channels * expansion;
//         bool has_se = reduction > 0;
//
//         if (expansion != 1) {
//             expand_conv = register_module("expand_conv", torch::nn::Conv2d(
//                 torch::nn::Conv2dOptions(in_channels, expanded_channels, 1).bias(false)));
//             bn0 = register_module("bn0", torch::nn::BatchNorm2d(expanded_channels));
//         }
//
//         depthwise_conv = register_module("depthwise_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(expanded_channels, expanded_channels, kernel_size)
//                 .stride(stride).padding(kernel_size / 2).groups(expanded_channels).bias(false)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(expanded_channels));
//
//         if (has_se) {
//             se = register_module("se", SEBlock(expanded_channels, reduction));
//         }
//
//         pointwise_conv = register_module("pointwise_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(expanded_channels, out_channels, 1).bias(false)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
//
//         skip_connection = (in_channels == out_channels && stride == 1);
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto out = x;
//         if (expand_conv) {
//             out = swish(bn0->forward(expand_conv->forward(out)));
//         }
//         out = swish(bn1->forward(depthwise_conv->forward(out)));
//         if (se) {
//             out = se->forward(out);
//         }
//         out = bn2->forward(pointwise_conv->forward(out));
//         if (skip_connection) {
//             out += x; // Residual connection
//         }
//         return out;
//     }
//
//     torch::nn::Conv2d expand_conv{nullptr}, depthwise_conv{nullptr}, pointwise_conv{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr}, bn2{nullptr};
//     SEBlock se{nullptr};
//     bool skip_connection;
// };
// TORCH_MODULE(MBConvBlock);
//
// // EfficientNetB1
// struct EfficientNetB1Impl : torch::nn::Module {
//     EfficientNetB1Impl(int num_classes = 10) {
//         // Initial stem
//         stem_conv = register_module("stem_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, 32, 3).stride(1).padding(1).bias(false))); // Simplified stride
//         bn0 = register_module("bn0", torch::nn::BatchNorm2d(32));
//
//         // MBConv blocks configuration: {num_repeats, in_channels, out_channels, expansion, kernel_size, stride, se_reduction}
//         std::vector<std::tuple<int, int, int, int, int, int, int>> config = {
//             {1, 32, 16, 1, 3, 1, 4},   // Stage 1
//             {2, 16, 24, 6, 3, 2, 4},   // Stage 2
//             {2, 24, 40, 6, 5, 2, 4},   // Stage 3
//             {3, 40, 80, 6, 3, 2, 4},   // Stage 4
//             {3, 80, 112, 6, 5, 1, 4},  // Stage 5
//             {4, 112, 192, 6, 5, 2, 4}, // Stage 6
//             {2, 192, 320, 6, 3, 1, 4}, // Stage 7 (increased repeats vs. B0)
//             {1, 320, 320, 6, 3, 1, 4}  // Stage 8 (extra stage vs. B0)
//         };
//
//         int stage_idx = 0;
//         for (const auto& [num_repeats, in_ch, out_ch, expansion, kernel, stride, reduction] : config) {
//             for (int i = 0; i < num_repeats; ++i) {
//                 int s = (i == 0) ? stride : 1;
//                 blocks->push_back(MBConvBlock(in_ch, out_ch, expansion, kernel, s, reduction));
//                 register_module("block_" + std::to_string(stage_idx) + "_" + std::to_string(i), blocks->back());
//                 in_ch = out_ch;
//             }
//             stage_idx++;
//         }
//
//         // Head
//         head_conv = register_module("head_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(320, 1280, 1).bias(false)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(1280));
//         fc = register_module("fc", torch::nn::Linear(1280, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = swish(bn0->forward(stem_conv->forward(x))); // [batch, 32, 32, 32]
//         for (auto& block : *blocks) {
//             x = block->forward(x);
//         }
//         x = swish(bn1->forward(head_conv->forward(x)));
//         x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, 1280, 1, 1]
//         x = x.view({x.size(0), -1}); // [batch, 1280]
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
//     torch::nn::Linear fc{nullptr};
//     torch::nn::ModuleList blocks{torch::nn::ModuleList()};
// };
// TORCH_MODULE(EfficientNetB1);
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
//         for (int c = 0; c < 3; ++c) {
//             std::getline(file, line);
//             std::istringstream iss(line);
//             for (int i = 0; i < image_size_ * image_size_; ++i) {
//                 iss >> image[c * image_size_ * image_size_ + i];
//             }
//         }
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
//
//         // Initialize model
//         EfficientNetB1 model(num_classes);
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
//                 torch::save(model, "efficientnetb1_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: efficientnetb1_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "efficientnetb1_final.pt");
//         std::cout << "Saved final model: efficientnetb1_final.pt" << std::endl;
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
// // Swish activation (x * sigmoid(x))
// torch::Tensor swish(torch::Tensor x) {
//     return x * torch::sigmoid(x);
// }
//
// // Squeeze-and-Excitation Block
// struct SEBlockImpl : torch::nn::Module {
//     SEBlockImpl(int in_channels, int reduction) {
//         fc1 = register_module("fc1", torch::nn::Linear(in_channels, in_channels / reduction));
//         fc2 = register_module("fc2", torch::nn::Linear(in_channels / reduction, in_channels));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto batch = x.size(0);
//         auto channels = x.size(1);
//         auto avg = torch::avg_pool2d(x, {x.size(2), x.size(3)}).view({batch, channels});
//         auto out = torch::relu(fc1->forward(avg));
//         out = torch::sigmoid(fc2->forward(out)).view({batch, channels, 1, 1});
//         return x * out;
//     }
//
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr};
// };
// TORCH_MODULE(SEBlock);
//
// // MBConv Block (Inverted Residual with Depthwise Separable Conv)
// struct MBConvBlockImpl : torch::nn::Module {
//     MBConvBlockImpl(int in_channels, int out_channels, int expansion, int kernel_size, int stride, int reduction) {
//         int expanded_channels = in_channels * expansion;
//         bool has_se = reduction > 0;
//
//         if (expansion != 1) {
//             expand_conv = register_module("expand_conv", torch::nn::Conv2d(
//                 torch::nn::Conv2dOptions(in_channels, expanded_channels, 1).bias(false)));
//             bn0 = register_module("bn0", torch::nn::BatchNorm2d(expanded_channels));
//         }
//
//         depthwise_conv = register_module("depthwise_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(expanded_channels, expanded_channels, kernel_size)
//                 .stride(stride).padding(kernel_size / 2).groups(expanded_channels).bias(false)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(expanded_channels));
//
//         if (has_se) {
//             se = register_module("se", SEBlock(expanded_channels, reduction));
//         }
//
//         pointwise_conv = register_module("pointwise_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(expanded_channels, out_channels, 1).bias(false)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
//
//         skip_connection = (in_channels == out_channels && stride == 1);
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto out = x;
//         if (expand_conv) {
//             out = swish(bn0->forward(expand_conv->forward(out)));
//         }
//         out = swish(bn1->forward(depthwise_conv->forward(out)));
//         if (se) {
//             out = se->forward(out);
//         }
//         out = bn2->forward(pointwise_conv->forward(out));
//         if (skip_connection) {
//             out += x; // Residual connection
//         }
//         return out;
//     }
//
//     torch::nn::Conv2d expand_conv{nullptr}, depthwise_conv{nullptr}, pointwise_conv{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr}, bn2{nullptr};
//     SEBlock se{nullptr};
//     bool skip_connection;
// };
// TORCH_MODULE(MBConvBlock);
//
// // EfficientNetB2
// struct EfficientNetB2Impl : torch::nn::Module {
//     EfficientNetB2Impl(int num_classes = 10) {
//         // Initial stem
//         stem_conv = register_module("stem_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, 32, 3).stride(1).padding(1).bias(false))); // Simplified stride
//         bn0 = register_module("bn0", torch::nn::BatchNorm2d(32));
//
//         // MBConv blocks configuration: {num_repeats, in_channels, out_channels, expansion, kernel_size, stride, se_reduction}
//         std::vector<std::tuple<int, int, int, int, int, int, int>> config = {
//             {2, 32, 16, 1, 3, 1, 4},   // Stage 1 (increased repeats vs. B1)
//             {3, 16, 24, 6, 3, 2, 4},   // Stage 2 (increased repeats vs. B1)
//             {3, 24, 48, 6, 5, 2, 4},   // Stage 3 (increased out_channels vs. B1)
//             {4, 48, 88, 6, 3, 2, 4},   // Stage 4 (increased repeats and out_channels vs. B1)
//             {4, 88, 120, 6, 5, 1, 4},  // Stage 5 (increased out_channels vs. B1)
//             {5, 120, 208, 6, 5, 2, 4}, // Stage 6 (increased repeats and out_channels vs. B1)
//             {2, 208, 352, 6, 3, 1, 4}, // Stage 7 (increased out_channels vs. B1)
//             {1, 352, 352, 6, 3, 1, 4}  // Stage 8
//         };
//
//         int stage_idx = 0;
//         for (const auto& [num_repeats, in_ch, out_ch, expansion, kernel, stride, reduction] : config) {
//             for (int i = 0; i < num_repeats; ++i) {
//                 int s = (i == 0) ? stride : 1;
//                 blocks->push_back(MBConvBlock(in_ch, out_ch, expansion, kernel, s, reduction));
//                 register_module("block_" + std::to_string(stage_idx) + "_" + std::to_string(i), blocks->back());
//                 in_ch = out_ch;
//             }
//             stage_idx++;
//         }
//
//         // Head
//         head_conv = register_module("head_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(352, 1408, 1).bias(false))); // Increased channels vs. B1
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(1408));
//         fc = register_module("fc", torch::nn::Linear(1408, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = swish(bn0->forward(stem_conv->forward(x))); // [batch, 32, 32, 32]
//         for (auto& block : *blocks) {
//             x = block->forward(x);
//         }
//         x = swish(bn1->forward(head_conv->forward(x)));
//         x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, 1408, 1, 1]
//         x = x.view({x.size(0), -1}); // [batch, 1408]
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
//     torch::nn::Linear fc{nullptr};
//     torch::nn::ModuleList blocks{torch::nn::ModuleList()};
// };
// TORCH_MODULE(EfficientNetB2);
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
//         for (int c = 0; c < 3; ++c) {
//             std::getline(file, line);
//             std::istringstream iss(line);
//             for (int i = 0; i < image_size_ * image_size_; ++i) {
//                 iss >> image[c * image_size_ * image_size_ + i];
//             }
//         }
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
//
//         // Initialize model
//         EfficientNetB2 model(num_classes);
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
//                 torch::save(model, "efficientnetb2_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: efficientnetb2_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "efficientnetb2_final.pt");
//         std::cout << "Saved final model: efficientnetb2_final.pt" << std::endl;
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
// // Swish activation (x * sigmoid(x))
// torch::Tensor swish(torch::Tensor x) {
//     return x * torch::sigmoid(x);
// }
//
// // Squeeze-and-Excitation Block
// struct SEBlockImpl : torch::nn::Module {
//     SEBlockImpl(int in_channels, int reduction) {
//         fc1 = register_module("fc1", torch::nn::Linear(in_channels, in_channels / reduction));
//         fc2 = register_module("fc2", torch::nn::Linear(in_channels / reduction, in_channels));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto batch = x.size(0);
//         auto channels = x.size(1);
//         auto avg = torch::avg_pool2d(x, {x.size(2), x.size(3)}).view({batch, channels});
//         auto out = torch::relu(fc1->forward(avg));
//         out = torch::sigmoid(fc2->forward(out)).view({batch, channels, 1, 1});
//         return x * out;
//     }
//
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr};
// };
// TORCH_MODULE(SEBlock);
//
// // MBConv Block (Inverted Residual with Depthwise Separable Conv)
// struct MBConvBlockImpl : torch::nn::Module {
//     MBConvBlockImpl(int in_channels, int out_channels, int expansion, int kernel_size, int stride, int reduction) {
//         int expanded_channels = in_channels * expansion;
//         bool has_se = reduction > 0;
//
//         if (expansion != 1) {
//             expand_conv = register_module("expand_conv", torch::nn::Conv2d(
//                 torch::nn::Conv2dOptions(in_channels, expanded_channels, 1).bias(false)));
//             bn0 = register_module("bn0", torch::nn::BatchNorm2d(expanded_channels));
//         }
//
//         depthwise_conv = register_module("depthwise_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(expanded_channels, expanded_channels, kernel_size)
//                 .stride(stride).padding(kernel_size / 2).groups(expanded_channels).bias(false)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(expanded_channels));
//
//         if (has_se) {
//             se = register_module("se", SEBlock(expanded_channels, reduction));
//         }
//
//         pointwise_conv = register_module("pointwise_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(expanded_channels, out_channels, 1).bias(false)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
//
//         skip_connection = (in_channels == out_channels && stride == 1);
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto out = x;
//         if (expand_conv) {
//             out = swish(bn0->forward(expand_conv->forward(out)));
//         }
//         out = swish(bn1->forward(depthwise_conv->forward(out)));
//         if (se) {
//             out = se->forward(out);
//         }
//         out = bn2->forward(pointwise_conv->forward(out));
//         if (skip_connection) {
//             out += x; // Residual connection
//         }
//         return out;
//     }
//
//     torch::nn::Conv2d expand_conv{nullptr}, depthwise_conv{nullptr}, pointwise_conv{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr}, bn2{nullptr};
//     SEBlock se{nullptr};
//     bool skip_connection;
// };
// TORCH_MODULE(MBConvBlock);
//
// // EfficientNetB3
// struct EfficientNetB3Impl : torch::nn::Module {
//     EfficientNetB3Impl(int num_classes = 10) {
//         // Initial stem
//         stem_conv = register_module("stem_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, 40, 3).stride(1).padding(1).bias(false))); // Simplified stride, increased channels
//         bn0 = register_module("bn0", torch::nn::BatchNorm2d(40));
//
//         // MBConv blocks configuration: {num_repeats, in_channels, out_channels, expansion, kernel_size, stride, se_reduction}
//         std::vector<std::tuple<int, int, int, int, int, int, int>> config = {
//             {2, 40, 24, 1, 3, 1, 4},   // Stage 1
//             {3, 24, 32, 6, 3, 2, 4},   // Stage 2
//             {4, 32, 48, 6, 5, 2, 4},   // Stage 3
//             {4, 48, 96, 6, 3, 2, 4},   // Stage 4
//             {5, 96, 136, 6, 5, 1, 4},  // Stage 5
//             {6, 136, 232, 6, 5, 2, 4}, // Stage 6
//             {3, 232, 384, 6, 3, 1, 4}, // Stage 7
//             {1, 384, 384, 6, 3, 1, 4}  // Stage 8
//         };
//
//         int stage_idx = 0;
//         for (const auto& [num_repeats, in_ch, out_ch, expansion, kernel, stride, reduction] : config) {
//             for (int i = 0; i < num_repeats; ++i) {
//                 int s = (i == 0) ? stride : 1;
//                 blocks->push_back(MBConvBlock(in_ch, out_ch, expansion, kernel, s, reduction));
//                 register_module("block_" + std::to_string(stage_idx) + "_" + std::to_string(i), blocks->back());
//                 in_ch = out_ch;
//             }
//             stage_idx++;
//         }
//
//         // Head
//         head_conv = register_module("head_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(384, 1536, 1).bias(false))); // Increased channels vs. B2
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(1536));
//         fc = register_module("fc", torch::nn::Linear(1536, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = swish(bn0->forward(stem_conv->forward(x))); // [batch, 40, 32, 32]
//         for (auto& block : *blocks) {
//             x = block->forward(x);
//         }
//         x = swish(bn1->forward(head_conv->forward(x)));
//         x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, 1536, 1, 1]
//         x = x.view({x.size(0), -1}); // [batch, 1536]
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
//     torch::nn::Linear fc{nullptr};
//     torch::nn::ModuleList blocks{torch::nn::ModuleList()};
// };
// TORCH_MODULE(EfficientNetB3);
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
//         for (int c = 0; c < 3; ++c) {
//             std::getline(file, line);
//             std::istringstream iss(line);
//             for (int i = 0; i < image_size_ * image_size_; ++i) {
//                 iss >> image[c * image_size_ * image_size_ + i];
//             }
//         }
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
//
//         // Initialize model
//         EfficientNetB3 model(num_classes);
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
//                 torch::save(model, "efficientnetb3_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: efficientnetb3_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "efficientnetb3_final.pt");
//         std::cout << "Saved final model: efficientnetb3_final.pt" << std::endl;
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
// // Swish activation (x * sigmoid(x))
// torch::Tensor swish(torch::Tensor x) {
//     return x * torch::sigmoid(x);
// }
//
// // Squeeze-and-Excitation Block
// struct SEBlockImpl : torch::nn::Module {
//     SEBlockImpl(int in_channels, int reduction) {
//         fc1 = register_module("fc1", torch::nn::Linear(in_channels, in_channels / reduction));
//         fc2 = register_module("fc2", torch::nn::Linear(in_channels / reduction, in_channels));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto batch = x.size(0);
//         auto channels = x.size(1);
//         auto avg = torch::avg_pool2d(x, {x.size(2), x.size(3)}).view({batch, channels});
//         auto out = torch::relu(fc1->forward(avg));
//         out = torch::sigmoid(fc2->forward(out)).view({batch, channels, 1, 1});
//         return x * out;
//     }
//
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr};
// };
// TORCH_MODULE(SEBlock);
//
// // MBConv Block (Inverted Residual with Depthwise Separable Conv)
// struct MBConvBlockImpl : torch::nn::Module {
//     MBConvBlockImpl(int in_channels, int out_channels, int expansion, int kernel_size, int stride, int reduction) {
//         int expanded_channels = in_channels * expansion;
//         bool has_se = reduction > 0;
//
//         if (expansion != 1) {
//             expand_conv = register_module("expand_conv", torch::nn::Conv2d(
//                 torch::nn::Conv2dOptions(in_channels, expanded_channels, 1).bias(false)));
//             bn0 = register_module("bn0", torch::nn::BatchNorm2d(expanded_channels));
//         }
//
//         depthwise_conv = register_module("depthwise_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(expanded_channels, expanded_channels, kernel_size)
//                 .stride(stride).padding(kernel_size / 2).groups(expanded_channels).bias(false)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(expanded_channels));
//
//         if (has_se) {
//             se = register_module("se", SEBlock(expanded_channels, reduction));
//         }
//
//         pointwise_conv = register_module("pointwise_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(expanded_channels, out_channels, 1).bias(false)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
//
//         skip_connection = (in_channels == out_channels && stride == 1);
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto out = x;
//         if (expand_conv) {
//             out = swish(bn0->forward(expand_conv->forward(out)));
//         }
//         out = swish(bn1->forward(depthwise_conv->forward(out)));
//         if (se) {
//             out = se->forward(out);
//         }
//         out = bn2->forward(pointwise_conv->forward(out));
//         if (skip_connection) {
//             out += x; // Residual connection
//         }
//         return out;
//     }
//
//     torch::nn::Conv2d expand_conv{nullptr}, depthwise_conv{nullptr}, pointwise_conv{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr}, bn2{nullptr};
//     SEBlock se{nullptr};
//     bool skip_connection;
// };
// TORCH_MODULE(MBConvBlock);
//
// // EfficientNetB4
// struct EfficientNetB4Impl : torch::nn::Module {
//     EfficientNetB4Impl(int num_classes = 10) {
//         // Initial stem
//         stem_conv = register_module("stem_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, 48, 3).stride(1).padding(1).bias(false))); // Simplified stride, increased channels
//         bn0 = register_module("bn0", torch::nn::BatchNorm2d(48));
//
//         // MBConv blocks configuration: {num_repeats, in_channels, out_channels, expansion, kernel_size, stride, se_reduction}
//         std::vector<std::tuple<int, int, int, int, int, int, int>> config = {
//             {2, 48, 24, 1, 3, 1, 4},   // Stage 1
//             {4, 24, 32, 6, 3, 2, 4},   // Stage 2
//             {4, 32, 56, 6, 5, 2, 4},   // Stage 3
//             {6, 56, 112, 6, 3, 2, 4},  // Stage 4
//             {6, 112, 160, 6, 5, 1, 4}, // Stage 5
//             {8, 160, 272, 6, 5, 2, 4}, // Stage 6
//             {3, 272, 448, 6, 3, 1, 4}, // Stage 7
//             {1, 448, 448, 6, 3, 1, 4}  // Stage 8
//         };
//
//         int stage_idx = 0;
//         for (const auto& [num_repeats, in_ch, out_ch, expansion, kernel, stride, reduction] : config) {
//             for (int i = 0; i < num_repeats; ++i) {
//                 int s = (i == 0) ? stride : 1;
//                 blocks->push_back(MBConvBlock(in_ch, out_ch, expansion, kernel, s, reduction));
//                 register_module("block_" + std::to_string(stage_idx) + "_" + std::to_string(i), blocks->back());
//                 in_ch = out_ch;
//             }
//             stage_idx++;
//         }
//
//         // Head
//         head_conv = register_module("head_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(448, 1792, 1).bias(false))); // Increased channels vs. B3
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(1792));
//         fc = register_module("fc", torch::nn::Linear(1792, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = swish(bn0->forward(stem_conv->forward(x))); // [batch, 48, 32, 32]
//         for (auto& block : *blocks) {
//             x = block->forward(x);
//         }
//         x = swish(bn1->forward(head_conv->forward(x)));
//         x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, 1792, 1, 1]
//         x = x.view({x.size(0), -1}); // [batch, 1792]
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
//     torch::nn::Linear fc{nullptr};
//     torch::nn::ModuleList blocks{torch::nn::ModuleList()};
// };
// TORCH_MODULE(EfficientNetB4);
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
//         for (int c = 0; c < 3; ++c) {
//             std::getline(file, line);
//             std::istringstream iss(line);
//             for (int i = 0; i < image_size_ * image_size_; ++i) {
//                 iss >> image[c * image_size_ * image_size_ + i];
//             }
//         }
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
//
//         // Initialize model
//         EfficientNetB4 model(num_classes);
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
//                 torch::save(model, "efficientnetb4_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: efficientnetb4_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "efficientnetb4_final.pt");
//         std::cout << "Saved final model: efficientnetb4_final.pt" << std::endl;
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
// // Swish activation (x * sigmoid(x))
// torch::Tensor swish(torch::Tensor x) {
//     return x * torch::sigmoid(x);
// }
//
// // Squeeze-and-Excitation Block
// struct SEBlockImpl : torch::nn::Module {
//     SEBlockImpl(int in_channels, int reduction) {
//         fc1 = register_module("fc1", torch::nn::Linear(in_channels, in_channels / reduction));
//         fc2 = register_module("fc2", torch::nn::Linear(in_channels / reduction, in_channels));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto batch = x.size(0);
//         auto channels = x.size(1);
//         auto avg = torch::avg_pool2d(x, {x.size(2), x.size(3)}).view({batch, channels});
//         auto out = torch::relu(fc1->forward(avg));
//         out = torch::sigmoid(fc2->forward(out)).view({batch, channels, 1, 1});
//         return x * out;
//     }
//
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr};
// };
// TORCH_MODULE(SEBlock);
//
// // MBConv Block (Inverted Residual with Depthwise Separable Conv)
// struct MBConvBlockImpl : torch::nn::Module {
//     MBConvBlockImpl(int in_channels, int out_channels, int expansion, int kernel_size, int stride, int reduction) {
//         int expanded_channels = in_channels * expansion;
//         bool has_se = reduction > 0;
//
//         if (expansion != 1) {
//             expand_conv = register_module("expand_conv", torch::nn::Conv2d(
//                 torch::nn::Conv2dOptions(in_channels, expanded_channels, 1).bias(false)));
//             bn0 = register_module("bn0", torch::nn::BatchNorm2d(expanded_channels));
//         }
//
//         depthwise_conv = register_module("depthwise_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(expanded_channels, expanded_channels, kernel_size)
//                 .stride(stride).padding(kernel_size / 2).groups(expanded_channels).bias(false)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(expanded_channels));
//
//         if (has_se) {
//             se = register_module("se", SEBlock(expanded_channels, reduction));
//         }
//
//         pointwise_conv = register_module("pointwise_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(expanded_channels, out_channels, 1).bias(false)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
//
//         skip_connection = (in_channels == out_channels && stride == 1);
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto out = x;
//         if (expand_conv) {
//             out = swish(bn0->forward(expand_conv->forward(out)));
//         }
//         out = swish(bn1->forward(depthwise_conv->forward(out)));
//         if (se) {
//             out = se->forward(out);
//         }
//         out = bn2->forward(pointwise_conv->forward(out));
//         if (skip_connection) {
//             out += x; // Residual connection
//         }
//         return out;
//     }
//
//     torch::nn::Conv2d expand_conv{nullptr}, depthwise_conv{nullptr}, pointwise_conv{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr}, bn2{nullptr};
//     SEBlock se{nullptr};
//     bool skip_connection;
// };
// TORCH_MODULE(MBConvBlock);
//
// // EfficientNetB5
// struct EfficientNetB5Impl : torch::nn::Module {
//     EfficientNetB5Impl(int num_classes = 10) {
//         // Initial stem
//         stem_conv = register_module("stem_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, 48, 3).stride(1).padding(1).bias(false))); // Simplified stride, increased channels
//         bn0 = register_module("bn0", torch::nn::BatchNorm2d(48));
//
//         // MBConv blocks configuration: {num_repeats, in_channels, out_channels, expansion, kernel_size, stride, se_reduction}
//         std::vector<std::tuple<int, int, int, int, int, int, int>> config = {
//             {3, 48, 24, 1, 3, 1, 4},   // Stage 1
//             {5, 24, 40, 6, 3, 2, 4},   // Stage 2
//             {5, 40, 64, 6, 5, 2, 4},   // Stage 3
//             {7, 64, 128, 6, 3, 2, 4},  // Stage 4
//             {8, 128, 176, 6, 5, 1, 4}, // Stage 5
//             {9, 176, 304, 6, 5, 2, 4}, // Stage 6
//             {4, 304, 512, 6, 3, 1, 4}, // Stage 7
//             {2, 512, 512, 6, 3, 1, 4}  // Stage 8
//         };
//
//         int stage_idx = 0;
//         for (const auto& [num_repeats, in_ch, out_ch, expansion, kernel, stride, reduction] : config) {
//             for (int i = 0; i < num_repeats; ++i) {
//                 int s = (i == 0) ? stride : 1;
//                 blocks->push_back(MBConvBlock(in_ch, out_ch, expansion, kernel, s, reduction));
//                 register_module("block_" + std::to_string(stage_idx) + "_" + std::to_string(i), blocks->back());
//                 in_ch = out_ch;
//             }
//             stage_idx++;
//         }
//
//         // Head
//         head_conv = register_module("head_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(512, 2048, 1).bias(false))); // Increased channels vs. B4
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(2048));
//         fc = register_module("fc", torch::nn::Linear(2048, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = swish(bn0->forward(stem_conv->forward(x))); // [batch, 48, 32, 32]
//         for (auto& block : *blocks) {
//             x = block->forward(x);
//         }
//         x = swish(bn1->forward(head_conv->forward(x)));
//         x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, 2048, 1, 1]
//         x = x.view({x.size(0), -1}); // [batch, 2048]
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
//     torch::nn::Linear fc{nullptr};
//     torch::nn::ModuleList blocks{torch::nn::ModuleList()};
// };
// TORCH_MODULE(EfficientNetB5);
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
//         for (int c = 0; c < 3; ++c) {
//             std::getline(file, line);
//             std::istringstream iss(line);
//             for (int i = 0; i < image_size_ * image_size_; ++i) {
//                 iss >> image[c * image_size_ * image_size_ + i];
//             }
//         }
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
//
//         // Initialize model
//         EfficientNetB5 model(num_classes);
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
//                 torch::save(model, "efficientnetb5_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: efficientnetb5_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "efficientnetb5_final.pt");
//         std::cout << "Saved final model: efficientnetb5_final.pt" << std::endl;
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
// // Swish activation (x * sigmoid(x))
// torch::Tensor swish(torch::Tensor x) {
//     return x * torch::sigmoid(x);
// }
//
// // Squeeze-and-Excitation Block
// struct SEBlockImpl : torch::nn::Module {
//     SEBlockImpl(int in_channels, int reduction) {
//         fc1 = register_module("fc1", torch::nn::Linear(in_channels, in_channels / reduction));
//         fc2 = register_module("fc2", torch::nn::Linear(in_channels / reduction, in_channels));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto batch = x.size(0);
//         auto channels = x.size(1);
//         auto avg = torch::avg_pool2d(x, {x.size(2), x.size(3)}).view({batch, channels});
//         auto out = torch::relu(fc1->forward(avg));
//         out = torch::sigmoid(fc2->forward(out)).view({batch, channels, 1, 1});
//         return x * out;
//     }
//
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr};
// };
// TORCH_MODULE(SEBlock);
//
// // MBConv Block (Inverted Residual with Depthwise Separable Conv)
// struct MBConvBlockImpl : torch::nn::Module {
//     MBConvBlockImpl(int in_channels, int out_channels, int expansion, int kernel_size, int stride, int reduction) {
//         int expanded_channels = in_channels * expansion;
//         bool has_se = reduction > 0;
//
//         if (expansion != 1) {
//             expand_conv = register_module("expand_conv", torch::nn::Conv2d(
//                 torch::nn::Conv2dOptions(in_channels, expanded_channels, 1).bias(false)));
//             bn0 = register_module("bn0", torch::nn::BatchNorm2d(expanded_channels));
//         }
//
//         depthwise_conv = register_module("depthwise_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(expanded_channels, expanded_channels, kernel_size)
//                 .stride(stride).padding(kernel_size / 2).groups(expanded_channels).bias(false)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(expanded_channels));
//
//         if (has_se) {
//             se = register_module("se", SEBlock(expanded_channels, reduction));
//         }
//
//         pointwise_conv = register_module("pointwise_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(expanded_channels, out_channels, 1).bias(false)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
//
//         skip_connection = (in_channels == out_channels && stride == 1);
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto out = x;
//         if (expand_conv) {
//             out = swish(bn0->forward(expand_conv->forward(out)));
//         }
//         out = swish(bn1->forward(depthwise_conv->forward(out)));
//         if (se) {
//             out = se->forward(out);
//         }
//         out = bn2->forward(pointwise_conv->forward(out));
//         if (skip_connection) {
//             out += x; // Residual connection
//         }
//         return out;
//     }
//
//     torch::nn::Conv2d expand_conv{nullptr}, depthwise_conv{nullptr}, pointwise_conv{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr}, bn2{nullptr};
//     SEBlock se{nullptr};
//     bool skip_connection;
// };
// TORCH_MODULE(MBConvBlock);
//
// // EfficientNetB6
// struct EfficientNetB6Impl : torch::nn::Module {
//     EfficientNetB6Impl(int num_classes = 10) {
//         // Initial stem
//         stem_conv = register_module("stem_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, 56, 3).stride(1).padding(1).bias(false))); // Simplified stride, increased channels
//         bn0 = register_module("bn0", torch::nn::BatchNorm2d(56));
//
//         // MBConv blocks configuration: {num_repeats, in_channels, out_channels, expansion, kernel_size, stride, se_reduction}
//         std::vector<std::tuple<int, int, int, int, int, int, int>> config = {
//             {3, 56, 32, 1, 3, 1, 4},   // Stage 1
//             {5, 32, 40, 6, 3, 2, 4},   // Stage 2
//             {6, 40, 72, 6, 5, 2, 4},   // Stage 3
//             {8, 72, 144, 6, 3, 2, 4},  // Stage 4
//             {9, 144, 200, 6, 5, 1, 4}, // Stage 5
//             {11, 200, 344, 6, 5, 2, 4}, // Stage 6
//             {5, 344, 576, 6, 3, 1, 4}, // Stage 7
//             {2, 576, 576, 6, 3, 1, 4}  // Stage 8
//         };
//
//         int stage_idx = 0;
//         for (const auto& [num_repeats, in_ch, out_ch, expansion, kernel, stride, reduction] : config) {
//             for (int i = 0; i < num_repeats; ++i) {
//                 int s = (i == 0) ? stride : 1;
//                 blocks->push_back(MBConvBlock(in_ch, out_ch, expansion, kernel, s, reduction));
//                 register_module("block_" + std::to_string(stage_idx) + "_" + std::to_string(i), blocks->back());
//                 in_ch = out_ch;
//             }
//             stage_idx++;
//         }
//
//         // Head
//         head_conv = register_module("head_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(576, 2304, 1).bias(false))); // Increased channels vs. B5
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(2304));
//         fc = register_module("fc", torch::nn::Linear(2304, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = swish(bn0->forward(stem_conv->forward(x))); // [batch, 56, 32, 32]
//         for (auto& block : *blocks) {
//             x = block->forward(x);
//         }
//         x = swish(bn1->forward(head_conv->forward(x)));
//         x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, 2304, 1, 1]
//         x = x.view({x.size(0), -1}); // [batch, 2304]
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
//     torch::nn::Linear fc{nullptr};
//     torch::nn::ModuleList blocks{torch::nn::ModuleList()};
// };
// TORCH_MODULE(EfficientNetB6);
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
//         for (int c = 0; c < 3; ++c) {
//             std::getline(file, line);
//             std::istringstream iss(line);
//             for (int i = 0; i < image_size_ * image_size_; ++i) {
//                 iss >> image[c * image_size_ * image_size_ + i];
//             }
//         }
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
//
//         // Initialize model
//         EfficientNetB6 model(num_classes);
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
//                 torch::save(model, "efficientnetb6_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: efficientnetb6_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "efficientnetb6_final.pt");
//         std::cout << "Saved final model: efficientnetb6_final.pt" << std::endl;
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
//
//     return 0;
// }















namespace xt::models
{
    EfficientNetB0::EfficientNetB0(int num_classes, int in_channels)
    {
    }

    EfficientNetB0::EfficientNetB0(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void EfficientNetB0::reset()
    {
    }

    auto EfficientNetB0::forward(std::initializer_list<std::any> tensors) -> std::any
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






    EfficientNetB1::EfficientNetB1(int num_classes, int in_channels)
    {
    }

    EfficientNetB1::EfficientNetB1(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void EfficientNetB1::reset()
    {
    }

    auto EfficientNetB1::forward(std::initializer_list<std::any> tensors) -> std::any
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








    EfficientNetB2::EfficientNetB2(int num_classes, int in_channels)
    {
    }

    EfficientNetB2::EfficientNetB2(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void EfficientNetB2::reset()
    {
    }

    auto EfficientNetB2::forward(std::initializer_list<std::any> tensors) -> std::any
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












    EfficientNetB3::EfficientNetB3(int num_classes, int in_channels)
    {
    }

    EfficientNetB3::EfficientNetB3(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void EfficientNetB3::reset()
    {
    }

    auto EfficientNetB3::forward(std::initializer_list<std::any> tensors) -> std::any
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












    EfficientNetB4::EfficientNetB4(int num_classes, int in_channels)
    {
    }

    EfficientNetB4::EfficientNetB4(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void EfficientNetB4::reset()
    {
    }

    auto EfficientNetB4::forward(std::initializer_list<std::any> tensors) -> std::any
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











    EfficientNetB5::EfficientNetB5(int num_classes, int in_channels)
    {
    }

    EfficientNetB5::EfficientNetB5(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void EfficientNetB5::reset()
    {
    }

    auto EfficientNetB5::forward(std::initializer_list<std::any> tensors) -> std::any
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


    EfficientNetB6::EfficientNetB6(int num_classes, int in_channels)
    {
    }

    EfficientNetB6::EfficientNetB6(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void EfficientNetB6::reset()
    {
    }

    auto EfficientNetB6::forward(std::initializer_list<std::any> tensors) -> std::any
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





    EfficientNetB7::EfficientNetB7(int num_classes, int in_channels)
    {
    }

    EfficientNetB7::EfficientNetB7(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void EfficientNetB7::reset()
    {
    }

    auto EfficientNetB7::forward(std::initializer_list<std::any> tensors) -> std::any
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
