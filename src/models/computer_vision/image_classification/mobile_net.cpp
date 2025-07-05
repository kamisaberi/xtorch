#include "include/models/computer_vision/image_classification/mobilenet.h"





//MobileNetV1 GROK


// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // Depthwise Separable Convolution
// struct DepthwiseSeparableConvImpl : torch::nn::Module {
//     DepthwiseSeparableConvImpl(int in_channels, int out_channels, int stride) {
//         dw_conv = register_module("dw_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, in_channels, 3).stride(stride).padding(1).groups(in_channels)));
//         dw_bn = register_module("dw_bn", torch::nn::BatchNorm2d(in_channels));
//         pw_conv = register_module("pw_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(1)));
//         pw_bn = register_module("pw_bn", torch::nn::BatchNorm2d(out_channels));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(dw_bn->forward(dw_conv->forward(x))); // Depthwise
//         x = torch::relu(pw_bn->forward(pw_conv->forward(x))); // Pointwise
//         return x;
//     }
//
//     torch::nn::Conv2d dw_conv{nullptr}, pw_conv{nullptr};
//     torch::nn::BatchNorm2d dw_bn{nullptr}, pw_bn{nullptr};
// };
// TORCH_MODULE(DepthwiseSeparableConv);
//
// // Simplified MobileNetV1
// struct MobileNetV1Impl : torch::nn::Module {
//     MobileNetV1Impl(int in_channels, int num_classes) {
//         // Stem
//         stem_conv = register_module("stem_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 32, 3).stride(2).padding(1)));
//         stem_bn = register_module("stem_bn", torch::nn::BatchNorm2d(32));
//
//         // Depthwise separable convolution blocks
//         blocks = torch::nn::Sequential(
//             DepthwiseSeparableConv(32, 64, 1),   // [batch, 64, 16, 16]
//             DepthwiseSeparableConv(64, 128, 2),  // [batch, 128, 8, 8]
//             DepthwiseSeparableConv(128, 128, 1), // [batch, 128, 8, 8]
//             DepthwiseSeparableConv(128, 256, 2)  // [batch, 256, 4, 4]
//         );
//         register_module("blocks", blocks);
//
//         // Head
//         pool = register_module("pool", torch::nn::AdaptiveAvgPool2d(1));
//         fc = register_module("fc", torch::nn::Linear(256, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, in_channels, 32, 32]
//         x = torch::relu(stem_bn->forward(stem_conv->forward(x))); // [batch, 32, 16, 16]
//         x = blocks->forward(x); // [batch, 256, 4, 4]
//         x = pool->forward(x).view({x.size(0), -1}); // [batch, 256]
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d stem_conv{nullptr};
//     torch::nn::BatchNorm2d stem_bn{nullptr};
//     torch::nn::Sequential blocks{nullptr};
//     torch::nn::AdaptiveAvgPool2d pool{nullptr};
//     torch::nn::Linear fc{nullptr};
// };
// TORCH_MODULE(MobileNetV1);
//
// // Classification Dataset
// struct ClassificationDataset : torch::data::Dataset<ClassificationDataset> {
//     ClassificationDataset(const std::string& img_dir, const std::string& label_dir) {
//         for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
//             if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
//                 image_paths_.push_back(entry.path().string());
//                 std::string label_path = label_dir + "/" + entry.path().filename().string() + ".txt";
//                 label_paths_.push_back(label_path);
//             }
//         }
//     }
//
//     torch::data::Example<> get(size_t index) override {
//         // Load image
//         cv::Mat image = cv::imread(image_paths_[index % image_paths_.size()], cv::IMREAD_GRAYSCALE);
//         if (image.empty()) {
//             throw std::runtime_error("Failed to load image: " + image_paths_[index % image_paths_.size()]);
//         }
//         cv::resize(image, image, cv::Size(32, 32));
//         image.convertTo(image, CV_32F, 2.0 / 255.0, -1.0); // Normalize to [-1, 1]
//         torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols}, torch::kFloat32);
//
//         // Load label
//         std::ifstream label_file(label_paths_[index % label_paths_.size()]);
//         if (!label_file.is_open()) {
//             throw std::runtime_error("Failed to open label file: " + label_paths_[index % label_paths_.size()]);
//         }
//         int label;
//         label_file >> label;
//         torch::Tensor label_tensor = torch::tensor(label, torch::kInt64);
//
//         return {img_tensor, label_tensor};
//     }
//
//     torch::optional<size_t> size() const override {
//         return image_paths_.size();
//     }
//
//     std::vector<std::string> image_paths_, label_paths_;
// };
//
// int main() {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Hyperparameters
//         const int in_channels = 1;
//         const int num_classes = 2; // Binary classification
//         const int batch_size = 32;
//         const float lr = 0.001;
//         const int num_epochs = 10;
//
//         // Initialize model
//         MobileNetV1 model(in_channels, num_classes);
//         model->to(device);
//
//         // Optimizer
//         torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));
//
//         // Load dataset
//         auto dataset = ClassificationDataset("./data/images", "./data/labels")
//             .map(torch::data::transforms::Stack<>());
//         auto data_loader = torch::data::make_data_loader(
//             dataset, torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));
//
//         // Training loop
//         model->train();
//         for (int epoch = 0; epoch < num_epochs; ++epoch) {
//             float loss_avg = 0.0;
//             int batch_count = 0;
//
//             for (auto& batch : *data_loader) {
//                 auto images = batch.data.to(device);
//                 auto labels = batch.target.to(device);
//
//                 optimizer.zero_grad();
//                 auto output = model->forward(images);
//                 auto loss = torch::nn::functional::cross_entropy(output, labels);
//                 loss.backward();
//                 optimizer.step();
//
//                 loss_avg += loss.item<float>();
//                 batch_count++;
//             }
//
//             std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "] "
//                       << "Loss: " << loss_avg / batch_count << std::endl;
//
//             // Save model every 5 epochs
//             if ((epoch + 1) % 5 == 0) {
//                 torch::save(model, "mobilenetv1_epoch_" + std::to_string(epoch + 1) + ".pt");
//             }
//         }
//
//         // Save final model
//         torch::save(model, "mobilenetv1.pt");
//         std::cout << "Model saved as mobilenetv1.pt" << std::endl;
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
//
//     return 0;
// }



//MobileNetV2 GROK
// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // Depthwise Separable Convolution
// struct DepthwiseSeparableConvImpl : torch::nn::Module {
//     DepthwiseSeparableConvImpl(int in_channels, int out_channels, int stride) {
//         dw_conv = register_module("dw_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, in_channels, 3).stride(stride).padding(1).groups(in_channels)));
//         dw_bn = register_module("dw_bn", torch::nn::BatchNorm2d(in_channels));
//         pw_conv = register_module("pw_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(1)));
//         pw_bn = register_module("pw_bn", torch::nn::BatchNorm2d(out_channels));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(dw_bn->forward(dw_conv->forward(x))); // Depthwise
//         x = pw_bn->forward(pw_conv->forward(x)); // Pointwise (linear bottleneck, no ReLU)
//         return x;
//     }
//
//     torch::nn::Conv2d dw_conv{nullptr}, pw_conv{nullptr};
//     torch::nn::BatchNorm2d dw_bn{nullptr}, pw_bn{nullptr};
// };
// TORCH_MODULE(DepthwiseSeparableConv);
//
// // Inverted Residual Block
// struct InvertedResidualBlockImpl : torch::nn::Module {
//     InvertedResidualBlockImpl(int in_channels, int exp_channels, int out_channels, int stride) {
//         expand_conv = nullptr;
//         if (in_channels != exp_channels) {
//             expand_conv = register_module("expand_conv", torch::nn::Conv2d(
//                 torch::nn::Conv2dOptions(in_channels, exp_channels, 1).stride(1)));
//             expand_bn = register_module("expand_bn", torch::nn::BatchNorm2d(exp_channels));
//         }
//         dw_conv = register_module("dw_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(exp_channels, exp_channels, 3).stride(stride).padding(1).groups(exp_channels)));
//         dw_bn = register_module("dw_bn", torch::nn::BatchNorm2d(exp_channels));
//         project_conv = register_module("project_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(exp_channels, out_channels, 1).stride(1)));
//         project_bn = register_module("project_bn", torch::nn::BatchNorm2d(out_channels));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto residual = x;
//         // Expansion
//         if (expand_conv) {
//             x = torch::relu(expand_bn->forward(expand_conv->forward(x)));
//         }
//         // Depthwise
//         x = torch::relu(dw_bn->forward(dw_conv->forward(x)));
//         // Projection (linear bottleneck, no ReLU)
//         x = project_bn->forward(project_conv->forward(x));
//         // Residual connection if shapes match
//         if (x.sizes() == residual.sizes()) {
//             x = x + residual;
//         }
//         return x;
//     }
//
//     torch::nn::Conv2d expand_conv{nullptr}, dw_conv{nullptr}, project_conv{nullptr};
//     torch::nn::BatchNorm2d expand_bn{nullptr}, dw_bn{nullptr}, project_bn{nullptr};
// };
// TORCH_MODULE(InvertedResidualBlock);
//
// // Simplified MobileNetV2
// struct MobileNetV2Impl : torch::nn::Module {
//     MobileNetV2Impl(int in_channels, int num_classes) {
//         // Stem
//         stem_conv = register_module("stem_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 32, 3).stride(2).padding(1)));
//         stem_bn = register_module("stem_bn", torch::nn::BatchNorm2d(32));
//
//         // Blocks: {in_channels, exp_channels, out_channels, stride}
//         blocks = torch::nn::Sequential(
//             InvertedResidualBlock(32, 32, 16, 1),    // [batch, 16, 16, 16]
//             InvertedResidualBlock(16, 96, 24, 2),    // [batch, 24, 8, 8]
//             InvertedResidualBlock(24, 144, 24, 1),   // [batch, 24, 8, 8]
//             InvertedResidualBlock(24, 144, 32, 2)    // [batch, 32, 4, 4]
//         );
//         register_module("blocks", blocks);
//
//         // Head
//         head_conv = register_module("head_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(32, 128, 1).stride(1)));
//         head_bn = register_module("head_bn", torch::nn::BatchNorm2d(128));
//         pool = register_module("pool", torch::nn::AdaptiveAvgPool2d(1));
//         fc = register_module("fc", torch::nn::Linear(128, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, in_channels, 32, 32]
//         x = torch::relu(stem_bn->forward(stem_conv->forward(x))); // [batch, 32, 16, 16]
//         x = blocks->forward(x); // [batch, 32, 4, 4]
//         x = torch::relu(head_bn->forward(head_conv->forward(x))); // [batch, 128, 4, 4]
//         x = pool->forward(x).view({x.size(0), -1}); // [batch, 128]
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
//     torch::nn::BatchNorm2d stem_bn{nullptr}, head_bn{nullptr};
//     torch::nn::Sequential blocks{nullptr};
//     torch::nn::AdaptiveAvgPool2d pool{nullptr};
//     torch::nn::Linear fc{nullptr};
// };
// TORCH_MODULE(MobileNetV2);
//
// // Classification Dataset
// struct ClassificationDataset : torch::data::Dataset<ClassificationDataset> {
//     ClassificationDataset(const std::string& img_dir, const std::string& label_dir) {
//         for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
//             if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
//                 image_paths_.push_back(entry.path().string());
//                 std::string label_path = label_dir + "/" + entry.path().filename().string() + ".txt";
//                 label_paths_.push_back(label_path);
//             }
//         }
//     }
//
//     torch::data::Example<> get(size_t index) override {
//         // Load image
//         cv::Mat image = cv::imread(image_paths_[index % image_paths_.size()], cv::IMREAD_GRAYSCALE);
//         if (image.empty()) {
//             throw std::runtime_error("Failed to load image: " + image_paths_[index % image_paths_.size()]);
//         }
//         cv::resize(image, image, cv::Size(32, 32));
//         image.convertTo(image, CV_32F, 2.0 / 255.0, -1.0); // Normalize to [-1, 1]
//         torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols}, torch::kFloat32);
//
//         // Load label
//         std::ifstream label_file(label_paths_[index % label_paths_.size()]);
//         if (!label_file.is_open()) {
//             throw std::runtime_error("Failed to open label file: " + label_paths_[index % label_paths_.size()]);
//         }
//         int label;
//         label_file >> label;
//         torch::Tensor label_tensor = torch::tensor(label, torch::kInt64);
//
//         return {img_tensor, label_tensor};
//     }
//
//     torch::optional<size_t> size() const override {
//         return image_paths_.size();
//     }
//
//     std::vector<std::string> image_paths_, label_paths_;
// };
//
// int main() {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Hyperparameters
//         const int in_channels = 1;
//         const int num_classes = 2; // Binary classification
//         const int batch_size = 32;
//         const float lr = 0.001;
//         const int num_epochs = 10;
//
//         // Initialize model
//         MobileNetV2 model(in_channels, num_classes);
//         model->to(device);
//
//         // Optimizer
//         torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));
//
//         // Load dataset
//         auto dataset = ClassificationDataset("./data/images", "./data/labels")
//             .map(torch::data::transforms::Stack<>());
//         auto data_loader = torch::data::make_data_loader(
//             dataset, torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));
//
//         // Training loop
//         model->train();
//         for (int epoch = 0; epoch < num_epochs; ++epoch) {
//             float loss_avg = 0.0;
//             int batch_count = 0;
//
//             for (auto& batch : *data_loader) {
//                 auto images = batch.data.to(device);
//                 auto labels = batch.target.to(device);
//
//                 optimizer.zero_grad();
//                 auto output = model->forward(images);
//                 auto loss = torch::nn::functional::cross_entropy(output, labels);
//                 loss.backward();
//                 optimizer.step();
//
//                 loss_avg += loss.item<float>();
//                 batch_count++;
//             }
//
//             std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "] "
//                       << "Loss: " << loss_avg / batch_count << std::endl;
//
//             // Save model every 5 epochs
//             if ((epoch + 1) % 5 == 0) {
//                 torch::save(model, "mobilenetv2_epoch_" + std::to_string(epoch + 1) + ".pt");
//             }
//         }
//
//         // Save final model
//         torch::save(model, "mobilenetv2.pt");
//         std::cout << "Model saved as mobilenetv2.pt" << std::endl;
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
//
//     return 0;
// }

//MobileNetV3 GROK

// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // Squeeze-and-Excitation Module
// struct SEModuleImpl : torch::nn::Module {
//     SEModuleImpl(int in_channels, int reduction = 4) {
//         fc1 = register_module("fc1", torch::nn::Linear(in_channels, in_channels / reduction));
//         fc2 = register_module("fc2", torch::nn::Linear(in_channels / reduction, in_channels));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, in_channels, h, w]
//         auto avg_pool = torch::avg_pool2d(x, {x.size(2), x.size(3)}).squeeze(-1).squeeze(-1); // [batch, in_channels]
//         auto se = torch::relu(fc1->forward(avg_pool)); // [batch, in_channels/reduction]
//         se = torch::sigmoid(fc2->forward(se)).unsqueeze(-1).unsqueeze(-1); // [batch, in_channels, 1, 1]
//         return x * se; // Element-wise multiplication
//     }
//
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr};
// };



// TORCH_MODULE(SEModule);
//
// // Depthwise Separable Convolution
// struct DepthwiseSeparableConvImpl : torch::nn::Module {
//     DepthwiseSeparableConvImpl(int in_channels, int out_channels, int stride) {
//         dw_conv = register_module("dw_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, in_channels, 3).stride(stride).padding(1).groups(in_channels)));
//         dw_bn = register_module("dw_bn", torch::nn::BatchNorm2d(in_channels));
//         pw_conv = register_module("pw_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(1)));
//         pw_bn = register_module("pw_bn", torch::nn::BatchNorm2d(out_channels));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(dw_bn->forward(dw_conv->forward(x))); // Depthwise
//         x = torch::relu(pw_bn->forward(pw_conv->forward(x))); // Pointwise
//         return x;
//     }
//
//     torch::nn::Conv2d dw_conv{nullptr}, pw_conv{nullptr};
//     torch::nn::BatchNorm2d dw_bn{nullptr}, pw_bn{nullptr};
// };
// TORCH_MODULE(DepthwiseSeparableConv);
//
// // Inverted Residual Block
// struct InvertedResidualBlockImpl : torch::nn::Module {
//     InvertedResidualBlockImpl(int in_channels, int exp_channels, int out_channels, int stride, bool use_se)
//         : use_se_(use_se) {
//         expand_conv = nullptr;
//         if (in_channels != exp_channels) {
//             expand_conv = register_module("expand_conv", torch::nn::Conv2d(
//                 torch::nn::Conv2dOptions(in_channels, exp_channels, 1).stride(1)));
//             expand_bn = register_module("expand_bn", torch::nn::BatchNorm2d(exp_channels));
//         }
//         dw_conv = register_module("dw_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(exp_channels, exp_channels, 3).stride(stride).padding(1).groups(exp_channels)));
//         dw_bn = register_module("dw_bn", torch::nn::BatchNorm2d(exp_channels));
//         if (use_se) {
//             se = register_module("se", SEModule(exp_channels));
//         }
//         project_conv = register_module("project_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(exp_channels, out_channels, 1).stride(1)));
//         project_bn = register_module("project_bn", torch::nn::BatchNorm2d(out_channels));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto residual = x;
//         // Expansion
//         if (expand_conv) {
//             x = torch::relu(expand_bn->forward(expand_conv->forward(x)));
//         }
//         // Depthwise
//         x = torch::relu(dw_bn->forward(dw_conv->forward(x)));
//         // Squeeze-and-Excitation
//         if (use_se_) {
//             x = se->forward(x);
//         }
//         // Projection
//         x = project_bn->forward(project_conv->forward(x));
//         // Residual connection if shapes match
//         if (x.sizes() == residual.sizes()) {
//             x = x + residual;
//         }
//         return x;
//     }
//
//     bool use_se_;
//     torch::nn::Conv2d expand_conv{nullptr}, dw_conv{nullptr}, project_conv{nullptr};
//     torch::nn::BatchNorm2d expand_bn{nullptr}, dw_bn{nullptr}, project_bn{nullptr};
//     SEModule se{nullptr};
// };
// TORCH_MODULE(InvertedResidualBlock);
//
// // Simplified MobileNetV3-Small
// struct MobileNetV3Impl : torch::nn::Module {
//     MobileNetV3Impl(int in_channels, int num_classes) {
//         // Stem
//         stem_conv = register_module("stem_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 16, 3).stride(2).padding(1)));
//         stem_bn = register_module("stem_bn", torch::nn::BatchNorm2d(16));
//
//         // Blocks: {in_channels, exp_channels, out_channels, stride, use_se}
//         blocks = torch::nn::Sequential(
//             InvertedResidualBlock(16, 16, 16, 2, true),   // [batch, 16, 8, 8]
//             InvertedResidualBlock(16, 72, 24, 2, false),  // [batch, 24, 4, 4]
//             InvertedResidualBlock(24, 88, 24, 1, false),  // [batch, 24, 4, 4]
//             InvertedResidualBlock(24, 96, 40, 2, true)    // [batch, 40, 2, 2]
//         );
//         register_module("blocks", blocks);
//
//         // Head
//         head_conv = register_module("head_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(40, 128, 1).stride(1)));
//         head_bn = register_module("head_bn", torch::nn::BatchNorm2d(128));
//         pool = register_module("pool", torch::nn::AdaptiveAvgPool2d(1));
//         fc = register_module("fc", torch::nn::Linear(128, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, in_channels, 32, 32]
//         x = torch::relu(stem_bn->forward(stem_conv->forward(x))); // [batch, 16, 16, 16]
//         x = blocks->forward(x); // [batch, 40, 2, 2]
//         x = torch::relu(head_bn->forward(head_conv->forward(x))); // [batch, 128, 2, 2]
//         x = pool->forward(x).view({x.size(0), -1}); // [batch, 128]
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
//     torch::nn::BatchNorm2d stem_bn{nullptr}, head_bn{nullptr};
//     torch::nn::Sequential blocks{nullptr};
//     torch::nn::AdaptiveAvgPool2d pool{nullptr};
//     torch::nn::Linear fc{nullptr};
// };
// TORCH_MODULE(MobileNetV3);
//
// // Classification Dataset
// struct ClassificationDataset : torch::data::Dataset<ClassificationDataset> {
//     ClassificationDataset(const std::string& img_dir, const std::string& label_dir) {
//         for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
//             if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
//                 image_paths_.push_back(entry.path().string());
//                 std::string label_path = label_dir + "/" + entry.path().filename().string() + ".txt";
//                 label_paths_.push_back(label_path);
//             }
//         }
//     }
//
//     torch::data::Example<> get(size_t index) override {
//         // Load image
//         cv::Mat image = cv::imread(image_paths_[index % image_paths_.size()], cv::IMREAD_GRAYSCALE);
//         if (image.empty()) {
//             throw std::runtime_error("Failed to load image: " + image_paths_[index % image_paths_.size()]);
//         }
//         cv::resize(image, image, cv::Size(32, 32));
//         image.convertTo(image, CV_32F, 2.0 / 255.0, -1.0); // Normalize to [-1, 1]
//         torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols}, torch::kFloat32);
//
//         // Load label
//         std::ifstream label_file(label_paths_[index % label_paths_.size()]);
//         if (!label_file.is_open()) {
//             throw std::runtime_error("Failed to open label file: " + label_paths_[index % label_paths_.size()]);
//         }
//         int label;
//         label_file >> label;
//         torch::Tensor label_tensor = torch::tensor(label, torch::kInt64);
//
//         return {img_tensor, label_tensor};
//     }
//
//     torch::optional<size_t> size() const override {
//         return image_paths_.size();
//     }
//
//     std::vector<std::string> image_paths_, label_paths_;
// };
//
// int main() {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Hyperparameters
//         const int in_channels = 1;
//         const int num_classes = 2; // Binary classification
//         const int batch_size = 32;
//         const float lr = 0.001;
//         const int num_epochs = 10;
//
//         // Initialize model
//         MobileNetV3 model(in_channels, num_classes);
//         model->to(device);
//
//         // Optimizer
//         torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));
//
//         // Load dataset
//         auto dataset = ClassificationDataset("./data/images", "./data/labels")
//             .map(torch::data::transforms::Stack<>());
//         auto data_loader = torch::data::make_data_loader(
//             dataset, torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));
//
//         // Training loop
//         model->train();
//         for (int epoch = 0; epoch < num_epochs; ++epoch) {
//             float loss_avg = 0.0;
//             int batch_count = 0;
//
//             for (auto& batch : *data_loader) {
//                 auto images = batch.data.to(device);
//                 auto labels = batch.target.to(device);
//
//                 optimizer.zero_grad();
//                 auto output = model->forward(images);
//                 auto loss = torch::nn::functional::cross_entropy(output, labels);
//                 loss.backward();
//                 optimizer.step();
//
//                 loss_avg += loss.item<float>();
//                 batch_count++;
//             }
//
//             std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "] "
//                       << "Loss: " << loss_avg / batch_count << std::endl;
//
//             // Save model every 5 epochs
//             if ((epoch + 1) % 5 == 0) {
//                 torch::save(model, "mobilenetv3_epoch_" + std::to_string(epoch + 1) + ".pt");
//             }
//         }
//
//         // Save final model
//         torch::save(model, "mobilenetv3.pt");
//         std::cout << "Model saved as mobilenetv3.pt" << std::endl;
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
//
//     return 0;
// }


namespace xt::models {
    HSigmoid::HSigmoid() {
        this->relu6 = torch::nn::ReLU6(torch::nn::ReLU6Options(true));
    }

    torch::Tensor HSigmoid::forward(torch::Tensor x) {
        x = this->relu6(x + 3) / 6;
        return x;
    }

    // --------------------------------------------------------------------

    HSwish::HSwish() {
        this->relu6 = torch::nn::ReLU6(torch::nn::ReLU6Options(true));
    }

    torch::Tensor HSwish::forward(torch::Tensor x) {
        x = this->relu6(x + 3) / 6;
        return x;
    }

    // --------------------------------------------------------------------

     SqueezeExcite::SqueezeExcite(int input_channels, int squeeze) {
         this->SE = torch::nn::Sequential(
             torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(1)),
             torch::nn::Conv2d(
                 torch::nn::Conv2dOptions(input_channels, input_channels / squeeze, 1).stride(1).bias(false)),
             torch::nn::BatchNorm2d(input_channels / squeeze),
             torch::nn::ReLU(torch::nn::ReLUOptions(true)),
             torch::nn::Conv2d(
                 torch::nn::Conv2dOptions(input_channels / squeeze, input_channels, 1).stride(1).bias(false)),
             torch::nn::BatchNorm2d(input_channels),
             HSigmoid()
         );
     }

     torch::Tensor SqueezeExcite::forward(torch::Tensor x) {
         x = this->SE->forward(x);
         return x;
     }
//
//     // --------------------------------------------------------------------
//
     Bottleneck::Bottleneck(int input_channels, int kernel, int stride, int expansion, int output_channels,
                            torch::nn::Module activation, bool se) {
         // this->bottleneck = torch::nn::Sequential(
         //     torch::nn::Conv2d(
         //         torch::nn::Conv2dOptions(input_channels, expansion, 1).stride(1).bias(false)),
         //     torch::nn::BatchNorm2d(expansion),
         //     activation,
         //
         //
         //     torch::nn::Conv2d(
         //         torch::nn::Conv2dOptions(expansion, expansion, kernel).stride(stride).padding(kernel / 2).
         //         groups(expansion).bias(false)),
         //     torch::nn::BatchNorm2d(expansion),
         //     activation,
         //
         //     SqueezeExcite(expansion),
         //
         //
         //     torch::nn::Conv2d(
         //         torch::nn::Conv2dOptions(expansion, output_channels, 1).stride(1).bias(false)),
         //     torch::nn::BatchNorm2d(expansion),
         //     activation
         // );


         if (input_channels == output_channels && stride == 1) {
             this->downsample = torch::nn::Sequential();
         } else {
             this->downsample = torch::nn::Sequential(
                 torch::nn::Conv2d(
                     torch::nn::Conv2dOptions(input_channels, output_channels, 1).stride(stride).bias(false)),
                 torch::nn::BatchNorm2d(expansion)
             );
         }
     }

     torch::Tensor Bottleneck::forward(torch::Tensor x) {
         torch::Tensor residual = x;
         torch::Tensor output = this->bottleneck->forward(x);
         if (this->downsample->size() != 0) {
             residual = this->downsample->forward(output);
         }
         output = output + residual;
         return output;
     }
//
//
//     // --------------------------------------------------------------------
//
//     MobileNetV3::MobileNetV3(int input_channels, int num_classes, float dropout_prob) {
//         this->initial_conv = torch::nn::Sequential(
//             torch::nn::Conv2d(
//                 torch::nn::Conv2dOptions(input_channels, 16, 3).stride(2)),
//             torch::nn::BatchNorm2d(16),
//             HSwish()
//         );
//
//         this->bottlenecks = torch::nn::Sequential(
//             Bottleneck(16, 3, 1, 16, 16, static_cast<Module>(*torch::nn::ReLU(torch::nn::ReLUOptions(true)))),
//             Bottleneck(16, 3, 2, 64, 24, static_cast<Module>(*torch::nn::ReLU(torch::nn::ReLUOptions(true)))),
//             Bottleneck(24, 3, 1, 72, 24, static_cast<Module>(*torch::nn::ReLU(torch::nn::ReLUOptions(true)))),
//             Bottleneck(24, 5, 2, 72, 40, static_cast<Module>(*torch::nn::ReLU(torch::nn::ReLUOptions(true)))),
//             Bottleneck(40, 5, 1, 120, 40, static_cast<Module>(*torch::nn::ReLU(torch::nn::ReLUOptions(true)))),
//             Bottleneck(40, 5, 1, 120, 40, static_cast<Module>(*torch::nn::ReLU(torch::nn::ReLUOptions(true)))),
//             Bottleneck(40, 3, 2, 240, 80, static_cast<Module>(HSwish())),
//             Bottleneck(80, 3, 1, 200, 80, static_cast<Module>(HSwish())),
//             Bottleneck(80, 3, 1, 184, 80, static_cast<Module>(HSwish())),
//             Bottleneck(80, 3, 1, 184, 80, static_cast<Module>(HSwish())),
//             Bottleneck(80, 3, 1, 480, 112, static_cast<Module>(HSwish()), true),
//             Bottleneck(112, 3, 1, 672, 112, static_cast<Module>(HSwish()), true),
//             Bottleneck(112, 5, 2, 672, 160, static_cast<Module>(HSwish()), true),
//             Bottleneck(160, 5, 1, 960, 160, static_cast<Module>(HSwish()), true),
//             Bottleneck(160, 5, 1, 960, 160, static_cast<Module>(HSwish()), true)
//         );
//
//
//         this->final_conv = torch::nn::Sequential(
//             torch::nn::Conv2d(
//                 torch::nn::Conv2dOptions(160, 960, 1).stride(1).bias(false)),
//             torch::nn::BatchNorm2d(960),
//             HSwish()
//         );
//
//
//         this->pool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(2));
//
//
//         this->classifier = torch::nn::Sequential(
//             torch::nn::Linear(960, 1200),
//             HSwish(),
//             torch::nn::Dropout(torch::nn::DropoutOptions(dropout_prob).inplace(true)),
//             torch::nn::Linear(1200, num_classes)
//         );
//     }
//
//     torch::Tensor MobileNetV3::forward(torch::Tensor x) {
//         x = this->initial_conv->forward(x);
//         x = this->bottlenecks->forward(x);
//         x = this->final_conv->forward(x);
//         x = this->pool(x);
//         x = x.view({x.size(0), -1});
//         //        x = torch.flatten(x, 1)
//         x = this->classifier->forward(x);
//         return x;
//     }




}
