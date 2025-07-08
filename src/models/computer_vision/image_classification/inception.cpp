#include "include/models/computer_vision/image_classification/inception.h"


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
// // Inception Module
// struct InceptionModuleImpl : torch::nn::Module {
//     InceptionModuleImpl(int in_channels, int ch1x1, int ch3x3_reduce, int ch3x3, int ch5x5_reduce, int ch5x5, int pool_proj) {
//         // Branch 1: 1x1 conv
//         conv1x1 = register_module("conv1x1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, ch1x1, 1).bias(false)));
//         bn1x1 = register_module("bn1x1", torch::nn::BatchNorm2d(ch1x1));
//
//         // Branch 2: 1x1 conv -> 3x3 conv
//         conv3x3_reduce = register_module("conv3x3_reduce", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, ch3x3_reduce, 1).bias(false)));
//         bn3x3_reduce = register_module("bn3x3_reduce", torch::nn::BatchNorm2d(ch3x3_reduce));
//         conv3x3 = register_module("conv3x3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(ch3x3_reduce, ch3x3, 3).padding(1).bias(false)));
//         bn3x3 = register_module("bn3x3", torch::nn::BatchNorm2d(ch3x3));
//
//         // Branch 3: 1x1 conv -> 5x5 conv
//         conv5x5_reduce = register_module("conv5x5_reduce", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, ch5x5_reduce, 1).bias(false)));
//         bn5x5_reduce = register_module("bn5x5_reduce", torch::nn::BatchNorm2d(ch5x5_reduce));
//         conv5x5 = register_module("conv5x5", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(ch5x5_reduce, ch5x5, 5).padding(2).bias(false)));
//         bn5x5 = register_module("bn5x5", torch::nn::BatchNorm2d(ch5x5));
//
//         // Branch 4: 3x3 max pool -> 1x1 conv
//         pool = register_module("pool", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(3).stride(1).padding(1)));
//         pool_proj = register_module("pool_proj", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, pool_proj, 1).bias(false)));
//         bn_pool = register_module("bn_pool", torch::nn::BatchNorm2d(pool_proj));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Branch 1
//         auto branch1 = torch::relu(bn1x1->forward(conv1x1->forward(x)));
//
//         // Branch 2
//         auto branch2 = torch::relu(bn3x3_reduce->forward(conv3x3_reduce->forward(x)));
//         branch2 = torch::relu(bn3x3->forward(conv3x3->forward(branch2)));
//
//         // Branch 3
//         auto branch3 = torch::relu(bn5x5_reduce->forward(conv5x5_reduce->forward(x)));
//         branch3 = torch::relu(bn5x5->forward(conv5x5->forward(branch3)));
//
//         // Branch 4
//         auto branch4 = pool->forward(x);
//         branch4 = torch::relu(bn_pool->forward(pool_proj->forward(branch4)));
//
//         // Concatenate along channel dimension
//         return torch::cat({branch1, branch2, branch3, branch4}, 1);
//     }
//
//     torch::nn::Conv2d conv1x1{nullptr}, conv3x3_reduce{nullptr}, conv3x3{nullptr};
//     torch::nn::Conv2d conv5x5_reduce{nullptr}, conv5x5{nullptr}, pool_proj{nullptr};
//     torch::nn::BatchNorm2d bn1x1{nullptr}, bn3x3_reduce{nullptr}, bn3x3{nullptr};
//     torch::nn::BatchNorm2d bn5x5_reduce{nullptr}, bn5x5{nullptr}, bn_pool{nullptr};
//     torch::nn::MaxPool2d pool{nullptr};
// };
// TORCH_MODULE(InceptionModule);
//
// // InceptionV1 (GoogLeNet)
// struct InceptionV1Impl : torch::nn::Module {
//     InceptionV1Impl(int num_classes = 10) {
//         // Stem
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, 64, 7).stride(1).padding(3).bias(false))); // Simplified stride
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
//         pool1 = register_module("pool1", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 64, 1).bias(false)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(64));
//         conv3 = register_module("conv3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 192, 3).padding(1).bias(false)));
//         bn3 = register_module("bn3", torch::nn::BatchNorm2d(192));
//         pool2 = register_module("pool2", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
//
//         // Inception modules: {in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj}
//         inception3a = register_module("inception3a", InceptionModule(192, 64, 96, 128, 16, 32, 32));
//         inception3b = register_module("inception3b", InceptionModule(256, 128, 128, 192, 32, 96, 64));
//         pool3 = register_module("pool3", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
//         inception4a = register_module("inception4a", InceptionModule(480, 192, 96, 208, 16, 48, 64));
//         inception4b = register_module("inception4b", InceptionModule(512, 160, 112, 224, 24, 64, 64));
//         inception4c = register_module("inception4c", InceptionModule(512, 128, 128, 256, 24, 64, 64));
//         inception4d = register_module("inception4d", InceptionModule(512, 112, 144, 288, 32, 64, 64));
//         inception4e = register_module("inception4e", InceptionModule(528, 256, 160, 320, 32, 128, 128));
//         pool4 = register_module("pool4", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
//         inception5a = register_module("inception5a", InceptionModule(832, 256, 160, 320, 32, 128, 128));
//         inception5b = register_module("inception5b", InceptionModule(832, 384, 192, 384, 48, 128, 128));
//
//         // Head
//         avg_pool = register_module("avg_pool", torch::nn::AdaptiveAvgPool2d(
//             torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
//         dropout = register_module("dropout", torch::nn::Dropout(0.4));
//         fc = register_module("fc", torch::nn::Linear(1024, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Stem: [batch, 3, 32, 32]
//         x = torch::relu(bn1->forward(conv1->forward(x))); // [batch, 64, 32, 32]
//         x = pool1->forward(x); // [batch, 64, 16, 16]
//         x = torch::relu(bn2->forward(conv2->forward(x))); // [batch, 64, 16, 16]
//         x = torch::relu(bn3->forward(conv3->forward(x))); // [batch, 192, 16, 16]
//         x = pool2->forward(x); // [batch, 192, 8, 8]
//
//         // Inception modules
//         x = inception3a->forward(x); // [batch, 256, 8, 8]
//         x = inception3b->forward(x); // [batch, 480, 8, 8]
//         x = pool3->forward(x); // [batch, 480, 4, 4]
//         x = inception4a->forward(x); // [batch, 512, 4, 4]
//         x = inception4b->forward(x); // [batch, 512, 4, 4]
//         x = inception4c->forward(x); // [batch, 512, 4, 4]
//         x = inception4d->forward(x); // [batch, 528, 4, 4]
//         x = inception4e->forward(x); // [batch, 832, 4, 4]
//         x = pool4->forward(x); // [batch, 832, 2, 2]
//         x = inception5a->forward(x); // [batch, 832, 2, 2]
//         x = inception5b->forward(x); // [batch, 1024, 2, 2]
//
//         // Head
//         x = avg_pool->forward(x); // [batch, 1024, 1, 1]
//         x = x.view({x.size(0), -1}); // [batch, 1024]
//         x = dropout->forward(x);
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
//     torch::nn::MaxPool2d pool1{nullptr}, pool2{nullptr}, pool3{nullptr}, pool4{nullptr};
//     torch::nn::AdaptiveAvgPool2d avg_pool{nullptr};
//     torch::nn::Dropout dropout{nullptr};
//     torch::nn::Linear fc{nullptr};
//     InceptionModule inception3a{nullptr}, inception3b{nullptr}, inception4a{nullptr};
//     InceptionModule inception4b{nullptr}, inception4c{nullptr}, inception4d{nullptr};
//     InceptionModule inception4e{nullptr}, inception5a{nullptr}, inception5b{nullptr};
// };
// TORCH_MODULE(InceptionV1);
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
//         InceptionV1 model(num_classes);
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
//                 torch::save(model, "inceptionv1_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: inceptionv1_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "inceptionv1_final.pt");
//         std::cout << "Saved final model: inceptionv1_final.pt" << std::endl;
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
// // Inception-A Module (Basic Inception block with factorized convolutions)
// struct InceptionAModuleImpl : torch::nn::Module {
//     InceptionAModuleImpl(int in_channels, int pool_features) {
//         // Branch 1: 1x1 conv
//         branch1x1 = register_module("branch1x1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 64, 1).bias(false)));
//         bn1x1 = register_module("bn1x1", torch::nn::BatchNorm2d(64));
//
//         // Branch 2: 1x1 conv -> 3x3 conv
//         branch3x3_1 = register_module("branch3x3_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 48, 1).bias(false)));
//         bn3x3_1 = register_module("bn3x3_1", torch::nn::BatchNorm2d(48));
//         branch3x3_2 = register_module("branch3x3_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(48, 64, 3).padding(1).bias(false)));
//         bn3x3_2 = register_module("bn3x3_2", torch::nn::BatchNorm2d(64));
//
//         // Branch 3: 1x1 conv -> 3x3 conv -> 3x3 conv
//         branch3x3dbl_1 = register_module("branch3x3dbl_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 64, 1).bias(false)));
//         bn3x3dbl_1 = register_module("bn3x3dbl_1", torch::nn::BatchNorm2d(64));
//         branch3x3dbl_2 = register_module("branch3x3dbl_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 96, 3).padding(1).bias(false)));
//         bn3x3dbl_2 = register_module("bn3x3dbl_2", torch::nn::BatchNorm2d(96));
//         branch3x3dbl_3 = register_module("branch3x3dbl_3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(96, 96, 3).padding(1).bias(false)));
//         bn3x3dbl_3 = register_module("bn3x3dbl_3", torch::nn::BatchNorm2d(96));
//
//         // Branch 4: Avg pool -> 1x1 conv
//         branch_pool = register_module("branch_pool", torch::nn::AvgPool2d(
//             torch::nn::AvgPool2dOptions(3).stride(1).padding(1)));
//         branch_pool_conv = register_module("branch_pool_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, pool_features, 1).bias(false)));
//         bn_pool = register_module("bn_pool", torch::nn::BatchNorm2d(pool_features));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Branch 1
//         auto branch1 = torch::relu(bn1x1->forward(branch1x1->forward(x)));
//
//         // Branch 2
//         auto branch2 = torch::relu(bn3x3_1->forward(branch3x3_1->forward(x)));
//         branch2 = torch::relu(bn3x3_2->forward(branch3x3_2->forward(branch2)));
//
//         // Branch 3
//         auto branch3 = torch::relu(bn3x3dbl_1->forward(branch3x3dbl_1->forward(x)));
//         branch3 = torch::relu(bn3x3dbl_2->forward(branch3x3dbl_2->forward(branch3)));
//         branch3 = torch::relu(bn3x3dbl_3->forward(branch3x3dbl_3->forward(branch3)));
//
//         // Branch 4
//         auto branch4 = branch_pool->forward(x);
//         branch4 = torch::relu(bn_pool->forward(branch_pool_conv->forward(branch4)));
//
//         // Concatenate along channel dimension
//         return torch::cat({branch1, branch2, branch3, branch4}, 1);
//     }
//
//     torch::nn::Conv2d branch1x1{nullptr}, branch3x3_1{nullptr}, branch3x3_2{nullptr};
//     torch::nn::Conv2d branch3x3dbl_1{nullptr}, branch3x3dbl_2{nullptr}, branch3x3dbl_3{nullptr};
//     torch::nn::Conv2d branch_pool_conv{nullptr};
//     torch::nn::BatchNorm2d bn1x1{nullptr}, bn3x3_1{nullptr}, bn3x3_2{nullptr};
//     torch::nn::BatchNorm2d bn3x3dbl_1{nullptr}, bn3x3dbl_2{nullptr}, bn3x3dbl_3{nullptr};
//     torch::nn::BatchNorm2d bn_pool{nullptr};
//     torch::nn::AvgPool2d branch_pool{nullptr};
// };
// TORCH_MODULE(InceptionAModule);
//
// // Inception-B Module (Grid size reduction with factorized convolutions)
// struct InceptionBModuleImpl : torch::nn::Module {
//     InceptionBModuleImpl(int in_channels) {
//         // Branch 1: 3x3 max pool
//         branch_pool = register_module("branch_pool", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
//
//         // Branch 2: 1x1 conv -> 3x3 conv -> 3x3 conv (stride 2)
//         branch3x3_1 = register_module("branch3x3_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 64, 1).bias(false)));
//         bn3x3_1 = register_module("bn3x3_1", torch::nn::BatchNorm2d(64));
//         branch3x3_2 = register_module("branch3x3_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 96, 3).padding(1).bias(false)));
//         bn3x3_2 = register_module("bn3x3_2", torch::nn::BatchNorm2d(96));
//         branch3x3_3 = register_module("branch3x3_3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(96, 96, 3).stride(2).bias(false)));
//         bn3x3_3 = register_module("bn3x3_3", torch::nn::BatchNorm2d(96));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Branch 1
//         auto branch1 = branch_pool->forward(x);
//
//         // Branch 2
//         auto branch2 = torch::relu(bn3x3_1->forward(branch3x3_1->forward(x)));
//         branch2 = torch::relu(bn3x3_2->forward(branch3x3_2->forward(branch2)));
//         branch2 = torch::relu(bn3x3_3->forward(branch3x3_3->forward(branch2)));
//
//         // Concatenate along channel dimension
//         return torch::cat({branch1, branch2}, 1);
//     }
//
//     torch::nn::MaxPool2d branch_pool{nullptr};
//     torch::nn::Conv2d branch3x3_1{nullptr}, branch3x3_2{nullptr}, branch3x3_3{nullptr};
//     torch::nn::BatchNorm2d bn3x3_1{nullptr}, bn3x3_2{nullptr}, bn3x3_3{nullptr};
// };
// TORCH_MODULE(InceptionBModule);
//
// // Inception-C Module (Asymmetric convolutions: nx1 and 1xn)
// struct InceptionCModuleImpl : torch::nn::Module {
//     InceptionCModuleImpl(int in_channels, int channels_7x7) {
//         // Branch 1: 1x1 conv
//         branch1x1 = register_module("branch1x1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 192, 1).bias(false)));
//         bn1x1 = register_module("bn1x1", torch::nn::BatchNorm2d(192));
//
//         // Branch 2: 1x1 conv -> 1x7 conv -> 7x1 conv
//         branch7x7_1 = register_module("branch7x7_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, channels_7x7, 1).bias(false)));
//         bn7x7_1 = register_module("bn7x7_1", torch::nn::BatchNorm2d(channels_7x7));
//         branch7x7_2 = register_module("branch7x7_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(channels_7x7, channels_7x7, {1, 7}).padding({0, 3}).bias(false)));
//         bn7x7_2 = register_module("bn7x7_2", torch::nn::BatchNorm2d(channels_7x7));
//         branch7x7_3 = register_module("branch7x7_3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(channels_7x7, 192, {7, 1}).padding({3, 0}).bias(false)));
//         bn7x7_3 = register_module("bn7x7_3", torch::nn::BatchNorm2d(192));
//
//         // Branch 3: 1x1 conv -> 1x7 conv -> 7x1 conv -> 1x7 conv -> 7x1 conv
//         branch7x7dbl_1 = register_module("branch7x7dbl_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, channels_7x7, 1).bias(false)));
//         bn7x7dbl_1 = register_module("bn7x7dbl_1", torch::nn::BatchNorm2d(channels_7x7));
//         branch7x7dbl_2 = register_module("branch7x7dbl_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(channels_7x7, channels_7x7, {1, 7}).padding({0, 3}).bias(false)));
//         bn7x7dbl_2 = register_module("bn7x7dbl_2", torch::nn::BatchNorm2d(channels_7x7));
//         branch7x7dbl_3 = register_module("branch7x7dbl_3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(channels_7x7, channels_7x7, {7, 1}).padding({3, 0}).bias(false)));
//         bn7x7dbl_3 = register_module("bn7x7dbl_3", torch::nn::BatchNorm2d(channels_7x7));
//         branch7x7dbl_4 = register_module("branch7x7dbl_4", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(channels_7x7, channels_7x7, {1, 7}).padding({0, 3}).bias(false)));
//         bn7x7dbl_4 = register_module("bn7x7dbl_4", torch::nn::BatchNorm2d(channels_7x7));
//         branch7x7dbl_5 = register_module("branch7x7dbl_5", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(channels_7x7, 192, {7, 1}).padding({3, 0}).bias(false)));
//         bn7x7dbl_5 = register_module("bn7x7dbl_5", torch::nn::BatchNorm2d(192));
//
//         // Branch 4: Avg pool -> 1x1 conv
//         branch_pool = register_module("branch_pool", torch::nn::AvgPool2d(
//             torch::nn::AvgPool2dOptions(3).stride(1).padding(1)));
//         branch_pool_conv = register_module("branch_pool_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 192, 1).bias(false)));
//         bn_pool = register_module("bn_pool", torch::nn::BatchNorm2d(192));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Branch 1
//         auto branch1 = torch::relu(bn1x1->forward(branch1x1->forward(x)));
//
//         // Branch 2
//         auto branch2 = torch::relu(bn7x7_1->forward(branch7x7_1->forward(x)));
//         branch2 = torch::relu(bn7x7_2->forward(branch7x7_2->forward(branch2)));
//         branch2 = torch::relu(bn7x7_3->forward(branch7x7_3->forward(branch2)));
//
//         // Branch 3
//         auto branch3 = torch::relu(bn7x7dbl_1->forward(branch7x7dbl_1->forward(x)));
//         branch3 = torch::relu(bn7x7dbl_2->forward(branch7x7dbl_2->forward(branch3)));
//         branch3 = torch::relu(bn7x7dbl_3->forward(branch7x7dbl_3->forward(branch3)));
//         branch3 = torch::relu(bn7x7dbl_4->forward(branch7x7dbl_4->forward(branch3)));
//         branch3 = torch::relu(bn7x7dbl_5->forward(branch7x7dbl_5->forward(branch3)));
//
//         // Branch 4
//         auto branch4 = branch_pool->forward(x);
//         branch4 = torch::relu(bn_pool->forward(branch_pool_conv->forward(branch4)));
//
//         // Concatenate along channel dimension
//         return torch::cat({branch1, branch2, branch3, branch4}, 1);
//     }
//
//     torch::nn::Conv2d branch1x1{nullptr}, branch7x7_1{nullptr}, branch7x7_2{nullptr}, branch7x7_3{nullptr};
//     torch::nn::Conv2d branch7x7dbl_1{nullptr}, branch7x7dbl_2{nullptr}, branch7x7dbl_3{nullptr};
//     torch::nn::Conv2d branch7x7dbl_4{nullptr}, branch7x7dbl_5{nullptr}, branch_pool_conv{nullptr};
//     torch::nn::BatchNorm2d bn1x1{nullptr}, bn7x7_1{nullptr}, bn7x7_2{nullptr}, bn7x7_3{nullptr};
//     torch::nn::BatchNorm2d bn7x7dbl_1{nullptr}, bn7x7dbl_2{nullptr}, bn7x7dbl_3{nullptr};
//     torch::nn::BatchNorm2d bn7x7dbl_4{nullptr}, bn7x7dbl_5{nullptr}, bn_pool{nullptr};
//     torch::nn::AvgPool2d branch_pool{nullptr};
// };
// TORCH_MODULE(InceptionCModule);
//
// // InceptionV2
// struct InceptionV2Impl : torch::nn::Module {
//     InceptionV2Impl(int num_classes = 10) {
//         // Stem
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, 32, 3).stride(1).padding(1).bias(false))); // Simplified stride
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(32, 32, 3).padding(1).bias(false)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(32));
//         conv3 = register_module("conv3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(32, 64, 3).padding(1).bias(false)));
//         bn3 = register_module("bn3", torch::nn::BatchNorm2d(64));
//         pool1 = register_module("pool1", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
//         conv4 = register_module("conv4", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 80, 1).bias(false)));
//         bn4 = register_module("bn4", torch::nn::BatchNorm2d(80));
//         conv5 = register_module("conv5", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(80, 192, 3).padding(1).bias(false)));
//         bn5 = register_module("bn5", torch::nn::BatchNorm2d(192));
//         pool2 = register_module("pool2", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
//
//         // Inception modules
//         inception3a = register_module("inception3a", InceptionAModule(192, 32));
//         inception3b = register_module("inception3b", InceptionAModule(256, 64));
//         inception3c = register_module("inception3c", InceptionAModule(320, 64));
//         inception4a = register_module("inception4a", InceptionBModule(384));
//         inception5a = register_module("inception5a", InceptionAModule(480, 128));
//         inception5b = register_module("inception5b", InceptionAModule(608, 128));
//         inception5c = register_module("inception5c", InceptionAModule(736, 128));
//         inception5d = register_module("inception5d", InceptionAModule(736, 128));
//         inception6a = register_module("inception6a", InceptionBModule(736));
//         inception7a = register_module("inception7a", InceptionCModule(832, 128));
//         inception7b = register_module("inception7b", InceptionCModule(832, 160));
//         inception7c = register_module("inception7c", InceptionCModule(832, 192));
//
//         // Head
//         avg_pool = register_module("avg_pool", torch::nn::AdaptiveAvgPool2d(
//             torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
//         dropout = register_module("dropout", torch::nn::Dropout(0.4));
//         fc = register_module("fc", torch::nn::Linear(768, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Stem: [batch, 3, 32, 32]
//         x = torch::relu(bn1->forward(conv1->forward(x))); // [batch, 32, 32, 32]
//         x = torch::relu(bn2->forward(conv2->forward(x))); // [batch, 32, 32, 32]
//         x = torch::relu(bn3->forward(conv3->forward(x))); // [batch, 64, 32, 32]
//         x = pool1->forward(x); // [batch, 64, 16, 16]
//         x = torch::relu(bn4->forward(conv4->forward(x))); // [batch, 80, 16, 16]
//         x = torch::relu(bn5->forward(conv5->forward(x))); // [batch, 192, 16, 16]
//         x = pool2->forward(x); // [batch, 192, 8, 8]
//
//         // Inception modules
//         x = inception3a->forward(x); // [batch, 256, 8, 8]
//         x = inception3b->forward(x); // [batch, 320, 8, 8]
//         x = inception3c->forward(x); // [batch, 384, 8, 8]
//         x = inception4a->forward(x); // [batch, 480, 4, 4]
//         x = inception5a->forward(x); // [batch, 608, 4, 4]
//         x = inception5b->forward(x); // [batch, 736, 4, 4]
//         x = inception5c->forward(x); // [batch, 736, 4, 4]
//         x = inception5d->forward(x); // [batch, 736, 4, 4]
//         x = inception6a->forward(x); // [batch, 832, 2, 2]
//         x = inception7a->forward(x); // [batch, 832, 2, 2]
//         x = inception7b->forward(x); // [batch, 832, 2, 2]
//         x = inception7c->forward(x); // [batch, 768, 2, 2]
//
//         // Head
//         x = avg_pool->forward(x); // [batch, 768, 1, 1]
//         x = x.view({x.size(0), -1}); // [batch, 768]
//         x = dropout->forward(x);
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr}, conv5{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr}, bn5{nullptr};
//     torch::nn::MaxPool2d pool1{nullptr}, pool2{nullptr};
//     torch::nn::AdaptiveAvgPool2d avg_pool{nullptr};
//     torch::nn::Dropout dropout{nullptr};
//     torch::nn::Linear fc{nullptr};
//     InceptionAModule inception3a{nullptr}, inception3b{nullptr}, inception3c{nullptr};
//     InceptionBModule inception4a{nullptr}, inception6a{nullptr};
//     InceptionAModule inception5a{nullptr}, inception5b{nullptr}, inception5c{nullptr}, inception5d{nullptr};
//     InceptionCModule inception7a{nullptr}, inception7b{nullptr}, inception7c{nullptr};
// };
// TORCH_MODULE(InceptionV2);
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
//         InceptionV2 model(num_classes);
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
//                 torch::save(model, "inceptionv2_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: inceptionv2_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "inceptionv2_final.pt");
//         std::cout << "Saved final model: inceptionv2_final.pt" << std::endl;
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
// // Inception-A Module
// struct InceptionAModuleImpl : torch::nn::Module {
//     InceptionAModuleImpl(int in_channels, int pool_features) {
//         // Branch 1: 1x1 conv
//         branch1x1 = register_module("branch1x1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 64, 1).bias(false)));
//         bn1x1 = register_module("bn1x1", torch::nn::BatchNorm2d(64));
//
//         // Branch 2: 1x1 conv -> 3x3 conv
//         branch3x3_1 = register_module("branch3x3_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 48, 1).bias(false)));
//         bn3x3_1 = register_module("bn3x3_1", torch::nn::BatchNorm2d(48));
//         branch3x3_2 = register_module("branch3x3_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(48, 64, 3).padding(1).bias(false)));
//         bn3x3_2 = register_module("bn3x3_2", torch::nn::BatchNorm2d(64));
//
//         // Branch 3: 1x1 conv -> 3x3 conv -> 3x3 conv
//         branch3x3dbl_1 = register_module("branch3x3dbl_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 64, 1).bias(false)));
//         bn3x3dbl_1 = register_module("bn3x3dbl_1", torch::nn::BatchNorm2d(64));
//         branch3x3dbl_2 = register_module("branch3x3dbl_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 96, 3).padding(1).bias(false)));
//         bn3x3dbl_2 = register_module("bn3x3dbl_2", torch::nn::BatchNorm2d(96));
//         branch3x3dbl_3 = register_module("branch3x3dbl_3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(96, 96, 3).padding(1).bias(false)));
//         bn3x3dbl_3 = register_module("bn3x3dbl_3", torch::nn::BatchNorm2d(96));
//
//         // Branch 4: Avg pool -> 1x1 conv
//         branch_pool = register_module("branch_pool", torch::nn::AvgPool2d(
//             torch::nn::AvgPool2dOptions(3).stride(1).padding(1)));
//         branch_pool_conv = register_module("branch_pool_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, pool_features, 1).bias(false)));
//         bn_pool = register_module("bn_pool", torch::nn::BatchNorm2d(pool_features));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto branch1 = torch::relu(bn1x1->forward(branch1x1->forward(x)));
//         auto branch2 = torch::relu(bn3x3_1->forward(branch3x3_1->forward(x)));
//         branch2 = torch::relu(bn3x3_2->forward(branch3x3_2->forward(branch2)));
//         auto branch3 = torch::relu(bn3x3dbl_1->forward(branch3x3dbl_1->forward(x)));
//         branch3 = torch::relu(bn3x3dbl_2->forward(branch3x3dbl_2->forward(branch3)));
//         branch3 = torch::relu(bn3x3dbl_3->forward(branch3x3dbl_3->forward(branch3)));
//         auto branch4 = branch_pool->forward(x);
//         branch4 = torch::relu(bn_pool->forward(branch_pool_conv->forward(branch4)));
//         return torch::cat({branch1, branch2, branch3, branch4}, 1);
//     }
//
//     torch::nn::Conv2d branch1x1{nullptr}, branch3x3_1{nullptr}, branch3x3_2{nullptr};
//     torch::nn::Conv2d branch3x3dbl_1{nullptr}, branch3x3dbl_2{nullptr}, branch3x3dbl_3{nullptr};
//     torch::nn::Conv2d branch_pool_conv{nullptr};
//     torch::nn::BatchNorm2d bn1x1{nullptr}, bn3x3_1{nullptr}, bn3x3_2{nullptr};
//     torch::nn::BatchNorm2d bn3x3dbl_1{nullptr}, bn3x3dbl_2{nullptr}, bn3x3dbl_3{nullptr};
//     torch::nn::BatchNorm2d bn_pool{nullptr};
//     torch::nn::AvgPool2d branch_pool{nullptr};
// };
// TORCH_MODULE(InceptionAModule);
//
// // Inception-B Module (Grid size reduction)
// struct InceptionBModuleImpl : torch::nn::Module {
//     InceptionBModuleImpl(int in_channels) {
//         // Branch 1: 3x3 max pool
//         branch_pool = register_module("branch_pool", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(3).stride(2)));
//
//         // Branch 2: 1x1 conv -> 3x3 conv -> 3x3 conv (stride 2)
//         branch3x3_1 = register_module("branch3x3_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 64, 1).bias(false)));
//         bn3x3_1 = register_module("bn3x3_1", torch::nn::BatchNorm2d(64));
//         branch3x3_2 = register_module("branch3x3_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 96, 3).padding(1).bias(false)));
//         bn3x3_2 = register_module("bn3x3_2", torch::nn::BatchNorm2d(96));
//         branch3x3_3 = register_module("branch3x3_3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(96, 96, 3).stride(2).bias(false)));
//         bn3x3_3 = register_module("bn3x3_3", torch::nn::BatchNorm2d(96));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto branch1 = branch_pool->forward(x);
//         auto branch2 = torch::relu(bn3x3_1->forward(branch3x3_1->forward(x)));
//         branch2 = torch::relu(bn3x3_2->forward(branch3x3_2->forward(branch2)));
//         branch2 = torch::relu(bn3x3_3->forward(branch3x3_3->forward(branch2)));
//         return torch::cat({branch1, branch2}, 1);
//     }
//
//     torch::nn::MaxPool2d branch_pool{nullptr};
//     torch::nn::Conv2d branch3x3_1{nullptr}, branch3x3_2{nullptr}, branch3x3_3{nullptr};
//     torch::nn::BatchNorm2d bn3x3_1{nullptr}, bn3x3_2{nullptr}, bn3x3_3{nullptr};
// };
// TORCH_MODULE(InceptionBModule);
//
// // Inception-C Module (Asymmetric factorized convolutions)
// struct InceptionCModuleImpl : torch::nn::Module {
//     InceptionCModuleImpl(int in_channels, int channels_7x7) {
//         // Branch 1: 1x1 conv
//         branch1x1 = register_module("branch1x1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 192, 1).bias(false)));
//         bn1x1 = register_module("bn1x1", torch::nn::BatchNorm2d(192));
//
//         // Branch 2: 1x1 conv -> 1x7 conv -> 7x1 conv
//         branch7x7_1 = register_module("branch7x7_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, channels_7x7, 1).bias(false)));
//         bn7x7_1 = register_module("bn7x7_1", torch::nn::BatchNorm2d(channels_7x7));
//         branch7x7_2 = register_module("branch7x7_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(channels_7x7, channels_7x7, {1, 7}).padding({0, 3}).bias(false)));
//         bn7x7_2 = register_module("bn7x7_2", torch::nn::BatchNorm2d(channels_7x7));
//         branch7x7_3 = register_module("branch7x7_3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(channels_7x7, 192, {7, 1}).padding({3, 0}).bias(false)));
//         bn7x7_3 = register_module("bn7x7_3", torch::nn::BatchNorm2d(192));
//
//         // Branch 3: 1x1 conv -> 1x7 conv -> 7x1 conv -> 1x7 conv -> 7x1 conv
//         branch7x7dbl_1 = register_module("branch7x7dbl_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, channels_7x7, 1).bias(false)));
//         bn7x7dbl_1 = register_module("bn7x7dbl_1", torch::nn::BatchNorm2d(channels_7x7));
//         branch7x7dbl_2 = register_module("branch7x7dbl_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(channels_7x7, channels_7x7, {1, 7}).padding({0, 3}).bias(false)));
//         bn7x7dbl_2 = register_module("bn7x7dbl_2", torch::nn::BatchNorm2d(channels_7x7));
//         branch7x7dbl_3 = register_module("branch7x7dbl_3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(channels_7x7, channels_7x7, {7, 1}).padding({3, 0}).bias(false)));
//         bn7x7dbl_3 = register_module("bn7x7dbl_3", torch::nn::BatchNorm2d(channels_7x7));
//         branch7x7dbl_4 = register_module("branch7x7dbl_4", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(channels_7x7, channels_7x7, {1, 7}).padding({0, 3}).bias(false)));
//         bn7x7dbl_4 = register_module("bn7x7dbl_4", torch::nn::BatchNorm2d(channels_7x7));
//         branch7x7dbl_5 = register_module("branch7x7dbl_5", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(channels_7x7, 192, {7, 1}).padding({3, 0}).bias(false)));
//         bn7x7dbl_5 = register_module("bn7x7dbl_5", torch::nn::BatchNorm2d(192));
//
//         // Branch 4: Avg pool -> 1x1 conv
//         branch_pool = register_module("branch_pool", torch::nn::AvgPool2d(
//             torch::nn::AvgPool2dOptions(3).stride(1).padding(1)));
//         branch_pool_conv = register_module("branch_pool_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 192, 1).bias(false)));
//         bn_pool = register_module("bn_pool", torch::nn::BatchNorm2d(192));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto branch1 = torch::relu(bn1x1->forward(branch1x1->forward(x)));
//         auto branch2 = torch::relu(bn7x7_1->forward(branch7x7_1->forward(x)));
//         branch2 = torch::relu(bn7x7_2->forward(branch7x7_2->forward(branch2)));
//         branch2 = torch::relu(bn7x7_3->forward(branch7x7_3->forward(branch2)));
//         auto branch3 = torch::relu(bn7x7dbl_1->forward(branch7x7dbl_1->forward(x)));
//         branch3 = torch::relu(bn7x7dbl_2->forward(branch7x7dbl_2->forward(branch3)));
//         branch3 = torch::relu(bn7x7dbl_3->forward(branch7x7dbl_3->forward(branch3)));
//         branch3 = torch::relu(bn7x7dbl_4->forward(branch7x7dbl_4->forward(branch3)));
//         branch3 = torch::relu(bn7x7dbl_5->forward(branch7x7dbl_5->forward(branch3)));
//         auto branch4 = branch_pool->forward(x);
//         branch4 = torch::relu(bn_pool->forward(branch_pool_conv->forward(branch4)));
//         return torch::cat({branch1, branch2, branch3, branch4}, 1);
//     }
//
//     torch::nn::Conv2d branch1x1{nullptr}, branch7x7_1{nullptr}, branch7x7_2{nullptr}, branch7x7_3{nullptr};
//     torch::nn::Conv2d branch7x7dbl_1{nullptr}, branch7x7dbl_2{nullptr}, branch7x7dbl_3{nullptr};
//     torch::nn::Conv2d branch7x7dbl_4{nullptr}, branch7x7dbl_5{nullptr}, branch_pool_conv{nullptr};
//     torch::nn::BatchNorm2d bn1x1{nullptr}, bn7x7_1{nullptr}, bn7x7_2{nullptr}, bn7x7_3{nullptr};
//     torch::nn::BatchNorm2d bn7x7dbl_1{nullptr}, bn7x7dbl_2{nullptr}, bn7x7dbl_3{nullptr};
//     torch::nn::BatchNorm2d bn7x7dbl_4{nullptr}, bn7x7dbl_5{nullptr}, bn_pool{nullptr};
//     torch::nn::AvgPool2d branch_pool{nullptr};
// };
// TORCH_MODULE(InceptionCModule);
//
// // Auxiliary Classifier
// struct AuxClassifierImpl : torch::nn::Module {
//     AuxClassifierImpl(int in_channels, int num_classes) {
//         conv = register_module("conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 128, 1).bias(false)));
//         bn = register_module("bn", torch::nn::BatchNorm2d(128));
//         fc1 = register_module("fc1", torch::nn::Linear(128 * 4 * 4, 1024));
//         fc2 = register_module("fc2", torch::nn::Linear(1024, num_classes));
//         pool = register_module("pool", torch::nn::AvgPool2d(
//             torch::nn::AvgPool2dOptions(5).stride(3)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = pool->forward(x);
//         x = torch::relu(bn->forward(conv->forward(x)));
//         x = x.view({x.size(0), -1});
//         x = torch::relu(fc1->forward(x));
//         x = fc2->forward(x);
//         return x;
//     }
//
//     torch::nn::Conv2d conv{nullptr};
//     torch::nn::BatchNorm2d bn{nullptr};
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr};
//     torch::nn::AvgPool2d pool{nullptr};
// };
// TORCH_MODULE(AuxClassifier);
//
// // InceptionV3
// struct InceptionV3Impl : torch::nn::Module {
//     InceptionV3Impl(int num_classes = 10, bool aux_logits = true) : aux_logits_(aux_logits) {
//         // Stem
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, 32, 3).stride(1).padding(1).bias(false))); // Simplified stride
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(32, 32, 3).padding(1).bias(false)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(32));
//         conv3 = register_module("conv3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(32, 64, 3).padding(1).bias(false)));
//         bn3 = register_module("bn3", torch::nn::BatchNorm2d(64));
//         pool1 = register_module("pool1", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
//         conv4 = register_module("conv4", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 80, 1).bias(false)));
//         bn4 = register_module("bn4", torch::nn::BatchNorm2d(80));
//         conv5 = register_module("conv5", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(80, 192, 3).padding(1).bias(false)));
//         bn5 = register_module("bn5", torch::nn::BatchNorm2d(192));
//         pool2 = register_module("pool2", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
//
//         // Inception modules
//         inception_a1 = register_module("inception_a1", InceptionAModule(192, 32));
//         inception_a2 = register_module("inception_a2", InceptionAModule(256, 64));
//         inception_a3 = register_module("inception_a3", InceptionAModule(288, 64));
//         inception_b = register_module("inception_b", InceptionBModule(288));
//         inception_c1 = register_module("inception_c1", InceptionAModule(768, 128));
//         inception_c2 = register_module("inception_c2", InceptionAModule(768, 128));
//         inception_c3 = register_module("inception_c3", InceptionAModule(768, 128));
//         inception_c4 = register_module("inception_c4", InceptionAModule(768, 128));
//         inception_d = register_module("inception_d", InceptionBModule(768));
//         inception_e1 = register_module("inception_e1", InceptionCModule(1280, 192));
//         inception_e2 = register_module("inception_e2", InceptionCModule(2048, 320));
//
//         // Auxiliary classifier
//         if (aux_logits_) {
//             aux_classifier = register_module("aux_classifier", AuxClassifier(768, num_classes));
//         }
//
//         // Head
//         avg_pool = register_module("avg_pool", torch::nn::AdaptiveAvgPool2d(
//             torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
//         dropout = register_module("dropout", torch::nn::Dropout(0.5));
//         fc = register_module("fc", torch::nn::Linear(2048, num_classes));
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
//         // Stem: [batch, 3, 32, 32]
//         x = torch::relu(bn1->forward(conv1->forward(x))); // [batch, 32, 32, 32]
//         x = torch::relu(bn2->forward(conv2->forward(x))); // [batch, 32, 32, 32]
//         x = torch::relu(bn3->forward(conv3->forward(x))); // [batch, 64, 32, 32]
//         x = pool1->forward(x); // [batch, 64, 16, 16]
//         x = torch::relu(bn4->forward(conv4->forward(x))); // [batch, 80, 16, 16]
//         x = torch::relu(bn5->forward(conv5->forward(x))); // [batch, 192, 16, 16]
//         x = pool2->forward(x); // [batch, 192, 8, 8]
//
//         // Inception-A
//         x = inception_a1->forward(x); // [batch, 256, 8, 8]
//         x = inception_a2->forward(x); // [batch, 288, 8, 8]
//         x = inception_a3->forward(x); // [batch, 288, 8, 8]
//
//         // Inception-B
//         x = inception_b->forward(x); // [batch, 768, 4, 4]
//
//         // Inception-C
//         x = inception_c1->forward(x); // [batch, 768, 4, 4]
//         x = inception_c2->forward(x); // [batch, 768, 4, 4]
//         x = inception_c3->forward(x); // [batch, 768, 4, 4]
//         torch::Tensor aux_output;
//         if (aux_logits_ && training()) {
//             aux_output = aux_classifier->forward(x);
//         }
//         x = inception_c4->forward(x); // [batch, 768, 4, 4]
//
//         // Inception-D
//         x = inception_d->forward(x); // [batch, 1280, 2, 2]
//
//         // Inception-E
//         x = inception_e1->forward(x); // [batch, 2048, 2, 2]
//         x = inception_e2->forward(x); // [batch, 2048, 2, 2]
//
//         // Head
//         x = avg_pool->forward(x); // [batch, 2048, 1, 1]
//         x = x.view({x.size(0), -1}); // [batch, 2048]
//         x = dropout->forward(x);
//         x = fc->forward(x); // [batch, num_classes]
//
//         return std::make_tuple(x, aux_output);
//     }
//
//     bool aux_logits_;
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr}, conv5{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr}, bn5{nullptr};
//     torch::nn::MaxPool2d pool1{nullptr}, pool2{nullptr};
//     torch::nn::AdaptiveAvgPool2d avg_pool{nullptr};
//     torch::nn::Dropout dropout{nullptr};
//     torch::nn::Linear fc{nullptr};
//     InceptionAModule inception_a1{nullptr}, inception_a2{nullptr}, inception_a3{nullptr};
//     InceptionBModule inception_b{nullptr}, inception_d{nullptr};
//     InceptionAModule inception_c1{nullptr}, inception_c2{nullptr}, inception_c3{nullptr}, inception_c4{nullptr};
//     InceptionCModule inception_e1{nullptr}, inception_e2{nullptr};
//     AuxClassifier aux_classifier{nullptr};
// };
// TORCH_MODULE(InceptionV3);
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
//         const float aux_loss_weight = 0.4;
//
//         // Initialize model
//         InceptionV3 model(num_classes, true);
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
//                 auto [output, aux_output] = model->forward(images);
//                 auto loss = torch::nn::functional::cross_entropy(output, labels);
//                 if (!aux_output.defined()) {
//                     std::cerr << "Auxiliary output is not defined!" << std::endl;
//                 } else {
//                     auto aux_loss = torch::nn::functional::cross_entropy(aux_output, labels);
//                     loss += aux_loss_weight * aux_loss;
//                 }
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
//                 torch::save(model, "inceptionv3_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: inceptionv3_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "inceptionv3_final.pt");
//         std::cout << "Saved final model: inceptionv3_final.pt" << std::endl;
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
// // Inception-A Module
// struct InceptionAModuleImpl : torch::nn::Module {
//     InceptionAModuleImpl(int in_channels) {
//         // Branch 1: 1x1 conv
//         branch1x1 = register_module("branch1x1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 96, 1).bias(false)));
//         bn1x1 = register_module("bn1x1", torch::nn::BatchNorm2d(96));
//
//         // Branch 2: 1x1 conv -> 3x3 conv
//         branch3x3_1 = register_module("branch3x3_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 64, 1).bias(false)));
//         bn3x3_1 = register_module("bn3x3_1", torch::nn::BatchNorm2d(64));
//         branch3x3_2 = register_module("branch3x3_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 96, 3).padding(1).bias(false)));
//         bn3x3_2 = register_module("bn3x3_2", torch::nn::BatchNorm2d(96));
//
//         // Branch 3: 1x1 conv -> 3x3 conv -> 3x3 conv
//         branch3x3dbl_1 = register_module("branch3x3dbl_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 64, 1).bias(false)));
//         bn3x3dbl_1 = register_module("bn3x3dbl_1", torch::nn::BatchNorm2d(64));
//         branch3x3dbl_2 = register_module("branch3x3dbl_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 96, 3).padding(1).bias(false)));
//         bn3x3dbl_2 = register_module("bn3x3dbl_2", torch::nn::BatchNorm2d(96));
//         branch3x3dbl_3 = register_module("branch3x3dbl_3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(96, 96, 3).padding(1).bias(false)));
//         bn3x3dbl_3 = register_module("bn3x3dbl_3", torch::nn::BatchNorm2d(96));
//
//         // Branch 4: Avg pool -> 1x1 conv
//         branch_pool = register_module("branch_pool", torch::nn::AvgPool2d(
//             torch::nn::AvgPool2dOptions(3).stride(1).padding(1)));
//         branch_pool_conv = register_module("branch_pool_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 96, 1).bias(false)));
//         bn_pool = register_module("bn_pool", torch::nn::BatchNorm2d(96));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto branch1 = torch::relu(bn1x1->forward(branch1x1->forward(x)));
//         auto branch2 = torch::relu(bn3x3_1->forward(branch3x3_1->forward(x)));
//         branch2 = torch::relu(bn3x3_2->forward(branch3x3_2->forward(branch2)));
//         auto branch3 = torch::relu(bn3x3dbl_1->forward(branch3x3dbl_1->forward(x)));
//         branch3 = torch::relu(bn3x3dbl_2->forward(branch3x3dbl_2->forward(branch3)));
//         branch3 = torch::relu(bn3x3dbl_3->forward(branch3x3dbl_3->forward(branch3)));
//         auto branch4 = branch_pool->forward(x);
//         branch4 = torch::relu(bn_pool->forward(branch_pool_conv->forward(branch4)));
//         return torch::cat({branch1, branch2, branch3, branch4}, 1);
//     }
//
//     torch::nn::Conv2d branch1x1{nullptr}, branch3x3_1{nullptr}, branch3x3_2{nullptr};
//     torch::nn::Conv2d branch3x3dbl_1{nullptr}, branch3x3dbl_2{nullptr}, branch3x3dbl_3{nullptr};
//     torch::nn::Conv2d branch_pool_conv{nullptr};
//     torch::nn::BatchNorm2d bn1x1{nullptr}, bn3x3_1{nullptr}, bn3x3_2{nullptr};
//     torch::nn::BatchNorm2d bn3x3dbl_1{nullptr}, bn3x3dbl_2{nullptr}, bn3x3dbl_3{nullptr};
//     torch::nn::BatchNorm2d bn_pool{nullptr};
//     torch::nn::AvgPool2d branch_pool{nullptr};
// };
// TORCH_MODULE(InceptionAModule);
//
// // Inception-B Module
// struct InceptionBModuleImpl : torch::nn::Module {
//     InceptionBModuleImpl(int in_channels) {
//         // Branch 1: 1x1 conv
//         branch1x1 = register_module("branch1x1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 384, 1).bias(false)));
//         bn1x1 = register_module("bn1x1", torch::nn::BatchNorm2d(384));
//
//         // Branch 2: 1x1 conv -> 1x7 conv -> 7x1 conv
//         branch7x7_1 = register_module("branch7x7_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 192, 1).bias(false)));
//         bn7x7_1 = register_module("bn7x7_1", torch::nn::BatchNorm2d(192));
//         branch7x7_2 = register_module("branch7x7_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(192, 224, {1, 7}).padding({0, 3}).bias(false)));
//         bn7x7_2 = register_module("bn7x7_2", torch::nn::BatchNorm2d(224));
//         branch7x7_3 = register_module("branch7x7_3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(224, 256, {7, 1}).padding({3, 0}).bias(false)));
//         bn7x7_3 = register_module("bn7x7_3", torch::nn::BatchNorm2d(256));
//
//         // Branch 3: 1x1 conv -> 1x7 conv -> 7x1 conv -> 1x7 conv -> 7x1 conv
//         branch7x7dbl_1 = register_module("branch7x7dbl_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 192, 1).bias(false)));
//         bn7x7dbl_1 = register_module("bn7x7dbl_1", torch::nn::BatchNorm2d(192));
//         branch7x7dbl_2 = register_module("branch7x7dbl_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(192, 192, {1, 7}).padding({0, 3}).bias(false)));
//         bn7x7dbl_2 = register_module("bn7x7dbl_2", torch::nn::BatchNorm2d(192));
//         branch7x7dbl_3 = register_module("branch7x7dbl_3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(192, 224, {7, 1}).padding({3, 0}).bias(false)));
//         bn7x7dbl_3 = register_module("bn7x7dbl_3", torch::nn::BatchNorm2d(224));
//         branch7x7dbl_4 = register_module("branch7x7dbl_4", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(224, 224, {1, 7}).padding({0, 3}).bias(false)));
//         bn7x7dbl_4 = register_module("bn7x7dbl_4", torch::nn::BatchNorm2d(224));
//         branch7x7dbl_5 = register_module("branch7x7dbl_5", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(224, 256, {7, 1}).padding({3, 0}).bias(false)));
//         bn7x7dbl_5 = register_module("bn7x7dbl_5", torch::nn::BatchNorm2d(256));
//
//         // Branch 4: Avg pool -> 1x1 conv
//         branch_pool = register_module("branch_pool", torch::nn::AvgPool2d(
//             torch::nn::AvgPool2dOptions(3).stride(1).padding(1)));
//         branch_pool_conv = register_module("branch_pool_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 128, 1).bias(false)));
//         bn_pool = register_module("bn_pool", torch::nn::BatchNorm2d(128));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto branch1 = torch::relu(bn1x1->forward(branch1x1->forward(x)));
//         auto branch2 = torch::relu(bn7x7_1->forward(branch7x7_1->forward(x)));
//         branch2 = torch::relu(bn7x7_2->forward(branch7x7_2->forward(branch2)));
//         branch2 = torch::relu(bn7x7_3->forward(branch7x7_3->forward(branch2)));
//         auto branch3 = torch::relu(bn7x7dbl_1->forward(branch7x7dbl_1->forward(x)));
//         branch3 = torch::relu(bn7x7dbl_2->forward(branch7x7dbl_2->forward(branch3)));
//         branch3 = torch::relu(bn7x7dbl_3->forward(branch7x7dbl_3->forward(branch3)));
//         branch3 = torch::relu(bn7x7dbl_4->forward(branch7x7dbl_4->forward(branch3)));
//         branch3 = torch::relu(bn7x7dbl_5->forward(branch7x7dbl_5->forward(branch3)));
//         auto branch4 = branch_pool->forward(x);
//         branch4 = torch::relu(bn_pool->forward(branch_pool_conv->forward(branch4)));
//         auto out = torch::cat({branch1, branch2, branch3, branch4}, 1);
//         return x + 0.3 * out; // Residual connection with scaling
//     }
//
//     torch::nn::Conv2d branch1x1{nullptr}, branch7x7_1{nullptr}, branch7x7_2{nullptr}, branch7x7_3{nullptr};
//     torch::nn::Conv2d branch7x7dbl_1{nullptr}, branch7x7dbl_2{nullptr}, branch7x7dbl_3{nullptr};
//     torch::nn::Conv2d branch7x7dbl_4{nullptr}, branch7x7dbl_5{nullptr}, branch_pool_conv{nullptr};
//     torch::nn::BatchNorm2d bn1x1{nullptr}, bn7x7_1{nullptr}, bn7x7_2{nullptr}, bn7x7_3{nullptr};
//     torch::nn::BatchNorm2d bn7x7dbl_1{nullptr}, bn7x7dbl_2{nullptr}, bn7x7dbl_3{nullptr};
//     torch::nn::BatchNorm2d bn7x7dbl_4{nullptr}, bn7x7dbl_5{nullptr}, bn_pool{nullptr};
//     torch::nn::AvgPool2d branch_pool{nullptr};
// };
// TORCH_MODULE(InceptionBModule);
//
// // Inception-C Module
// struct InceptionCModuleImpl : torch::nn::Module {
//     InceptionCModuleImpl(int in_channels) {
//         // Branch 1: 1x1 conv
//         branch1x1 = register_module("branch1x1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 256, 1).bias(false)));
//         bn1x1 = register_module("bn1x1", torch::nn::BatchNorm2d(256));
//
//         // Branch 2: 1x1 conv -> parallel 1x3 and 3x1 conv
//         branch3x3_1 = register_module("branch3x3_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 384, 1).bias(false)));
//         bn3x3_1 = register_module("bn3x3_1", torch::nn::BatchNorm2d(384));
//         branch3x3_2a = register_module("branch3x3_2a", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(384, 256, {1, 3}).padding({0, 1}).bias(false)));
//         bn3x3_2a = register_module("bn3x3_2a", torch::nn::BatchNorm2d(256));
//         branch3x3_2b = register_module("branch3x3_2b", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(384, 256, {3, 1}).padding({1, 0}).bias(false)));
//         bn3x3_2b = register_module("bn3x3_2b", torch::nn::BatchNorm2d(256));
//
//         // Branch 3: 1x1 conv -> 1x3 conv -> 3x1 conv -> parallel 1x3 and 3x1 conv
//         branch3x3dbl_1 = register_module("branch3x3dbl_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 384, 1).bias(false)));
//         bn3x3dbl_1 = register_module("bn3x3dbl_1", torch::nn::BatchNorm2d(384));
//         branch3x3dbl_2 = register_module("branch3x3dbl_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(384, 448, {1, 3}).padding({0, 1}).bias(false)));
//         bn3x3dbl_2 = register_module("bn3x3dbl_2", torch::nn::BatchNorm2d(448));
//         branch3x3dbl_3 = register_module("branch3x3dbl_3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(448, 512, {3, 1}).padding({1, 0}).bias(false)));
//         bn3x3dbl_3 = register_module("bn3x3dbl_3", torch::nn::BatchNorm2d(512));
//         branch3x3dbl_4a = register_module("branch3x3dbl_4a", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(512, 256, {1, 3}).padding({0, 1}).bias(false)));
//         bn3x3dbl_4a = register_module("bn3x3dbl_4a", torch::nn::BatchNorm2d(256));
//         branch3x3dbl_4b = register_module("branch3x3dbl_4b", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(512, 256, {3, 1}).padding({1, 0}).bias(false)));
//         bn3x3dbl_4b = register_module("bn3x3dbl_4b", torch::nn::BatchNorm2d(256));
//
//         // Branch 4: Avg pool -> 1x1 conv
//         branch_pool = register_module("branch_pool", torch::nn::AvgPool2d(
//             torch::nn::AvgPool2dOptions(3).stride(1).padding(1)));
//         branch_pool_conv = register_module("branch_pool_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 256, 1).bias(false)));
//         bn_pool = register_module("bn_pool", torch::nn::BatchNorm2d(256));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto branch1 = torch::relu(bn1x1->forward(branch1x1->forward(x)));
//         auto branch2 = torch::relu(bn3x3_1->forward(branch3x3_1->forward(x)));
//         auto branch2a = torch::relu(bn3x3_2a->forward(branch3x3_2a->forward(branch2)));
//         auto branch2b = torch::relu(bn3x3_2b->forward(branch3x3_2b->forward(branch2)));
//         auto branch3 = torch::relu(bn3x3dbl_1->forward(branch3x3dbl_1->forward(x)));
//         branch3 = torch::relu(bn3x3dbl_2->forward(branch3x3dbl_2->forward(branch3)));
//         branch3 = torch::relu(bn3x3dbl_3->forward(branch3x3dbl_3->forward(branch3)));
//         auto branch3a = torch::relu(bn3x3dbl_4a->forward(branch3x3dbl_4a->forward(branch3)));
//         auto branch3b = torch::relu(bn3x3dbl_4b->forward(branch3x3dbl_4b->forward(branch3)));
//         auto branch4 = branch_pool->forward(x);
//         branch4 = torch::relu(bn_pool->forward(branch_pool_conv->forward(branch4)));
//         auto out = torch::cat({branch1, branch2a, branch2b, branch3a, branch3b, branch4}, 1);
//         return x + 0.3 * out; // Residual connection with scaling
//     }
//
//     torch::nn::Conv2d branch1x1{nullptr}, branch3x3_1{nullptr}, branch3x3_2a{nullptr}, branch3x3_2b{nullptr};
//     torch::nn::Conv2d branch3x3dbl_1{nullptr}, branch3x3dbl_2{nullptr}, branch3x3dbl_3{nullptr};
//     torch::nn::Conv2d branch3x3dbl_4a{nullptr}, branch3x3dbl_4b{nullptr}, branch_pool_conv{nullptr};
//     torch::nn::BatchNorm2d bn1x1{nullptr}, bn3x3_1{nullptr}, bn3x3_2a{nullptr}, bn3x3_2b{nullptr};
//     torch::nn::BatchNorm2d bn3x3dbl_1{nullptr}, bn3x3dbl_2{nullptr}, bn3x3dbl_3{nullptr};
//     torch::nn::BatchNorm2d bn3x3dbl_4a{nullptr}, bn3x3dbl_4b{nullptr}, bn_pool{nullptr};
//     torch::nn::AvgPool2d branch_pool{nullptr};
// };
// TORCH_MODULE(InceptionCModule);
//
// // Reduction-A Module
// struct ReductionAModuleImpl : torch::nn::Module {
//     ReductionAModuleImpl(int in_channels, int k, int l, int m, int n) {
//         // Branch 1: 3x3 max pool
//         branch_pool = register_module("branch_pool", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(3).stride(2)));
//
//         // Branch 2: 3x3 conv (stride 2)
//         branch3x3 = register_module("branch3x3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, n, 3).stride(2).bias(false)));
//         bn3x3 = register_module("bn3x3", torch::nn::BatchNorm2d(n));
//
//         // Branch 3: 1x1 conv -> 3x3 conv -> 3x3 conv (stride 2)
//         branch3x3dbl_1 = register_module("branch3x3dbl_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, k, 1).bias(false)));
//         bn3x3dbl_1 = register_module("bn3x3dbl_1", torch::nn::BatchNorm2d(k));
//         branch3x3dbl_2 = register_module("branch3x3dbl_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(k, l, 3).padding(1).bias(false)));
//         bn3x3dbl_2 = register_module("bn3x3dbl_2", torch::nn::BatchNorm2d(l));
//         branch3x3dbl_3 = register_module("branch3x3dbl_3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(l, m, 3).stride(2).bias(false)));
//         bn3x3dbl_3 = register_module("bn3x3dbl_3", torch::nn::BatchNorm2d(m));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto branch1 = branch_pool->forward(x);
//         auto branch2 = torch::relu(bn3x3->forward(branch3x3->forward(x)));
//         auto branch3 = torch::relu(bn3x3dbl_1->forward(branch3x3dbl_1->forward(x)));
//         branch3 = torch::relu(bn3x3dbl_2->forward(branch3x3dbl_2->forward(branch3)));
//         branch3 = torch::relu(bn3x3dbl_3->forward(branch3x3dbl_3->forward(branch3)));
//         return torch::cat({branch1, branch2, branch3}, 1);
//     }
//
//     torch::nn::MaxPool2d branch_pool{nullptr};
//     torch::nn::Conv2d branch3x3{nullptr}, branch3x3dbl_1{nullptr}, branch3x3dbl_2{nullptr}, branch3x3dbl_3{nullptr};
//     torch::nn::BatchNorm2d bn3x3{nullptr}, bn3x3dbl_1{nullptr}, bn3x3dbl_2{nullptr}, bn3x3dbl_3{nullptr};
// };
// TORCH_MODULE(ReductionAModule);
//
// // Reduction-B Module
// struct ReductionBModuleImpl : torch::nn::Module {
//     ReductionBModuleImpl(int in_channels) {
//         // Branch 1: 3x3 max pool
//         branch_pool = register_module("branch_pool", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(3).stride(2)));
//
//         // Branch 2: 1x1 conv -> 3x3 conv (stride 2)
//         branch3x3_1 = register_module("branch3x3_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 192, 1).bias(false)));
//         bn3x3_1 = register_module("bn3x3_1", torch::nn::BatchNorm2d(192));
//         branch3x3_2 = register_module("branch3x3_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(192, 192, 3).stride(2).bias(false)));
//         bn3x3_2 = register_module("bn3x3_2", torch::nn::BatchNorm2d(192));
//
//         // Branch 3: 1x1 conv -> 1x7 conv -> 7x1 conv -> 3x3 conv (stride 2)
//         branch7x7_1 = register_module("branch7x7_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 256, 1).bias(false)));
//         bn7x7_1 = register_module("bn7x7_1", torch::nn::BatchNorm2d(256));
//         branch7x7_2 = register_module("branch7x7_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, {1, 7}).padding({0, 3}).bias(false)));
//         bn7x7_2 = register_module("bn7x7_2", torch::nn::BatchNorm2d(256));
//         branch7x7_3 = register_module("branch7x7_3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 320, {7, 1}).padding({3, 0}).bias(false)));
//         bn7x7_3 = register_module("bn7x7_3", torch::nn::BatchNorm2d(320));
//         branch7x7_4 = register_module("branch7x7_4", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(320, 320, 3).stride(2).bias(false)));
//         bn7x7_4 = register_module("bn7x7_4", torch::nn::BatchNorm2d(320));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto branch1 = branch_pool->forward(x);
//         auto branch2 = torch::relu(bn3x3_1->forward(branch3x3_1->forward(x)));
//         branch2 = torch::relu(bn3x3_2->forward(branch3x3_2->forward(branch2)));
//         auto branch3 = torch::relu(bn7x7_1->forward(branch7x7_1->forward(x)));
//         branch3 = torch::relu(bn7x7_2->forward(branch7x7_2->forward(branch3)));
//         branch3 = torch::relu(bn7x7_3->forward(branch7x7_3->forward(branch3)));
//         branch3 = torch::relu(bn7x7_4->forward(branch7x7_4->forward(branch3)));
//         return torch::cat({branch1, branch2, branch3}, 1);
//     }
//
//     torch::nn::MaxPool2d branch_pool{nullptr};
//     torch::nn::Conv2d branch3x3_1{nullptr}, branch3x3_2{nullptr};
//     torch::nn::Conv2d branch7x7_1{nullptr}, branch7x7_2{nullptr}, branch7x7_3{nullptr}, branch7x7_4{nullptr};
//     torch::nn::BatchNorm2d bn3x3_1{nullptr}, bn3x3_2{nullptr};
//     torch::nn::BatchNorm2d bn7x7_1{nullptr}, bn7x7_2{nullptr}, bn7x7_3{nullptr}, bn7x7_4{nullptr};
// };
// TORCH_MODULE(ReductionBModule);
//
// // InceptionV4
// struct InceptionV4Impl : torch::nn::Module {
//     InceptionV4Impl(int num_classes = 10) {
//         // Stem
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, 32, 3).stride(1).padding(1).bias(false)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(32, 32, 3).padding(1).bias(false)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(32));
//         conv3 = register_module("conv3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(32, 64, 3).padding(1).bias(false)));
//         bn3 = register_module("bn3", torch::nn::BatchNorm2d(64));
//         pool1 = register_module("pool1", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
//         conv4 = register_module("conv4", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 96, 3).padding(1).bias(false)));
//         bn4 = register_module("bn4", torch::nn::BatchNorm2d(96));
//         conv5 = register_module("conv5", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(96, 64, 1).bias(false)));
//         bn5 = register_module("bn5", torch::nn::BatchNorm2d(64));
//         conv6 = register_module("conv6", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 96, 3).padding(1).bias(false)));
//         bn6 = register_module("bn6", torch::nn::BatchNorm2d(96));
//         conv7 = register_module("conv7", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(160, 192, 3).stride(2).bias(false)));
//         bn7 = register_module("bn7", torch::nn::BatchNorm2d(192));
//
//         // Inception modules
//         inception_a1 = register_module("inception_a1", InceptionAModule(192));
//         inception_a2 = register_module("inception_a2", InceptionAModule(384));
//         inception_a3 = register_module("inception_a3", InceptionAModule(384));
//         inception_a4 = register_module("inception_a4", InceptionAModule(384));
//         reduction_a = register_module("reduction_a", ReductionAModule(384, 192, 224, 256, 384));
//         inception_b1 = register_module("inception_b1", InceptionBModule(1024));
//         inception_b2 = register_module("inception_b2", InceptionBModule(1024));
//         inception_b3 = register_module("inception_b3", InceptionBModule(1024));
//         inception_b4 = register_module("inception_b4", InceptionBModule(1024));
//         inception_b5 = register_module("inception_b5", InceptionBModule(1024));
//         inception_b6 = register_module("inception_b6", InceptionBModule(1024));
//         inception_b7 = register_module("inception_b7", InceptionBModule(1024));
//         reduction_b = register_module("reduction_b", ReductionBModule(1024));
//         inception_c1 = register_module("inception_c1", InceptionCModule(1536));
//         inception_c2 = register_module("inception_c2", InceptionCModule(1536));
//         inception_c3 = register_module("inception_c3", InceptionCModule(1536));
//
//         // Head
//         avg_pool = register_module("avg_pool", torch::nn::AdaptiveAvgPool2d(
//             torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
//         dropout = register_module("dropout", torch::nn::Dropout(0.8));
//         fc = register_module("fc", torch::nn::Linear(1536, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Stem: [batch, 3, 32, 32]
//         x = torch::relu(bn1->forward(conv1->forward(x))); // [batch, 32, 32, 32]
//         x = torch::relu(bn2->forward(conv2->forward(x))); // [batch, 32, 32, 32]
//         x = torch::relu(bn3->forward(conv3->forward(x))); // [batch, 64, 32, 32]
//         x = pool1->forward(x); // [batch, 64, 16, 16]
//         x = torch::relu(bn4->forward(conv4->forward(x))); // [batch, 96, 16, 16]
//         auto branch1 = torch::relu(bn5->forward(conv5->forward(x))); // [batch, 64, 16, 16]
//         auto branch2 = torch::relu(bn6->forward(conv6->forward(x))); // [batch, 96, 16, 16]
//         x = torch::cat({branch1, branch2}, 1); // [batch, 160, 16, 16]
//         x = torch::relu(bn7->forward(conv7->forward(x))); // [batch, 192, 8, 8]
//
//         // Inception-A
//         x = inception_a1->forward(x); // [batch, 384, 8, 8]
//         x = inception_a2->forward(x); // [batch, 384, 8, 8]
//         x = inception_a3->forward(x); // [batch, 384, 8, 8]
//         x = inception_a4->forward(x); // [batch, 384, 8, 8]
//
//         // Reduction-A
//         x = reduction_a->forward(x); // [batch, 1024, 4, 4]
//
//         // Inception-B
//         x = inception_b1->forward(x); // [batch, 1024, 4, 4]
//         x = inception_b2->forward(x); // [batch, 1024, 4, 4]
//         x = inception_b3->forward(x); // [batch, 1024, 4, 4]
//         x = inception_b4->forward(x); // [batch, 1024, 4, 4]
//         x = inception_b5->forward(x); // [batch, 1024, 4, 4]
//         x = inception_b6->forward(x); // [batch, 1024, 4, 4]
//         x = inception_b7->forward(x); // [batch, 1024, 4, 4]
//
//         // Reduction-B
//         x = reduction_b->forward(x); // [batch, 1536, 2, 2]
//
//         // Inception-C
//         x = inception_c1->forward(x); // [batch, 1536, 2, 2]
//         x = inception_c2->forward(x); // [batch, 1536, 2, 2]
//         x = inception_c3->forward(x); // [batch, 1536, 2, 2]
//
//         // Head
//         x = avg_pool->forward(x); // [batch, 1536, 1, 1]
//         x = x.view({x.size(0), -1}); // [batch, 1536]
//         x = dropout->forward(x);
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
//     torch::nn::Conv2d conv5{nullptr}, conv6{nullptr}, conv7{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr};
//     torch::nn::BatchNorm2d bn5{nullptr}, bn6{nullptr}, bn7{nullptr};
//     torch::nn::MaxPool2d pool1{nullptr};
//     torch::nn::AdaptiveAvgPool2d avg_pool{nullptr};
//     torch::nn::Dropout dropout{nullptr};
//     torch::nn::Linear fc{nullptr};
//     InceptionAModule inception_a1{nullptr}, inception_a2{nullptr}, inception_a3{nullptr}, inception_a4{nullptr};
//     ReductionAModule reduction_a{nullptr};
//     InceptionBModule inception_b1{nullptr}, inception_b2{nullptr}, inception_b3{nullptr};
//     InceptionBModule inception_b4{nullptr}, inception_b5{nullptr}, inception_b6{nullptr}, inception_b7{nullptr};
//     ReductionBModule reduction_b{nullptr};
//     InceptionCModule inception_c1{nullptr}, inception_c2{nullptr}, inception_c3{nullptr};
// };
// TORCH_MODULE(InceptionV4);
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
//         InceptionV4 model(num_classes);
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
//                 torch::save(model, "inceptionv4_epoch_" + std::to_string(epoch + 1) + ".pt");
//                 std::cout << "Saved checkpoint: inceptionv4_epoch_" << epoch + 1 << ".pt" << std::endl;
//             }
//         }
//
//         // Save final model
//         torch::save(model, "inceptionv4_final.pt");
//         std::cout << "Saved final model: inceptionv4_final.pt" << std::endl;
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
    // Inception Module
    InceptionModuleImpl::InceptionModuleImpl(int in_channels, int ch1x1, int ch3x3_reduce, int ch3x3, int ch5x5_reduce,
                                             int ch5x5,
                                             int pool_proj_out)
    {
        // Branch 1: 1x1 conv
        conv1x1 = register_module("conv1x1", torch::nn::Conv2d(
                                      torch::nn::Conv2dOptions(in_channels, ch1x1, 1).bias(false)));
        bn1x1 = register_module("bn1x1", torch::nn::BatchNorm2d(ch1x1));

        // Branch 2: 1x1 conv -> 3x3 conv
        conv3x3_reduce = register_module("conv3x3_reduce", torch::nn::Conv2d(
                                             torch::nn::Conv2dOptions(in_channels, ch3x3_reduce, 1).bias(false)));
        bn3x3_reduce = register_module("bn3x3_reduce", torch::nn::BatchNorm2d(ch3x3_reduce));
        conv3x3 = register_module("conv3x3", torch::nn::Conv2d(
                                      torch::nn::Conv2dOptions(ch3x3_reduce, ch3x3, 3).padding(1).bias(false)));
        bn3x3 = register_module("bn3x3", torch::nn::BatchNorm2d(ch3x3));

        // Branch 3: 1x1 conv -> 5x5 conv
        conv5x5_reduce = register_module("conv5x5_reduce", torch::nn::Conv2d(
                                             torch::nn::Conv2dOptions(in_channels, ch5x5_reduce, 1).bias(false)));
        bn5x5_reduce = register_module("bn5x5_reduce", torch::nn::BatchNorm2d(ch5x5_reduce));
        conv5x5 = register_module("conv5x5", torch::nn::Conv2d(
                                      torch::nn::Conv2dOptions(ch5x5_reduce, ch5x5, 5).padding(2).bias(false)));
        bn5x5 = register_module("bn5x5", torch::nn::BatchNorm2d(ch5x5));

        // Branch 4: 3x3 max pool -> 1x1 conv
        pool = register_module("pool", torch::nn::MaxPool2d(
                                   torch::nn::MaxPool2dOptions(3).stride(1).padding(1)));
        pool_proj = register_module("pool_proj", torch::nn::Conv2d(
                                        torch::nn::Conv2dOptions(in_channels, pool_proj_out, 1).bias(false)));
        bn_pool = register_module("bn_pool", torch::nn::BatchNorm2d(pool_proj_out));
    }

    torch::Tensor InceptionModuleImpl::forward(torch::Tensor x)
    {
        // Branch 1
        auto branch1 = torch::relu(bn1x1->forward(conv1x1->forward(x)));

        // Branch 2
        auto branch2 = torch::relu(bn3x3_reduce->forward(conv3x3_reduce->forward(x)));
        branch2 = torch::relu(bn3x3->forward(conv3x3->forward(branch2)));

        // Branch 3
        auto branch3 = torch::relu(bn5x5_reduce->forward(conv5x5_reduce->forward(x)));
        branch3 = torch::relu(bn5x5->forward(conv5x5->forward(branch3)));

        // Branch 4
        auto branch4 = pool->forward(x);
        branch4 = torch::relu(bn_pool->forward(pool_proj->forward(branch4)));

        // Concatenate along channel dimension
        return torch::cat({branch1, branch2, branch3, branch4}, 1);
    }


    InceptionAModuleImpl::InceptionAModuleImpl(int in_channels, int pool_features)
    {
        // Branch 1: 1x1 conv
        branch1x1 = register_module("branch1x1", torch::nn::Conv2d(
                                        torch::nn::Conv2dOptions(in_channels, 64, 1).bias(false)));
        bn1x1 = register_module("bn1x1", torch::nn::BatchNorm2d(64));

        // Branch 2: 1x1 conv -> 3x3 conv
        branch3x3_1 = register_module("branch3x3_1", torch::nn::Conv2d(
                                          torch::nn::Conv2dOptions(in_channels, 48, 1).bias(false)));
        bn3x3_1 = register_module("bn3x3_1", torch::nn::BatchNorm2d(48));
        branch3x3_2 = register_module("branch3x3_2", torch::nn::Conv2d(
                                          torch::nn::Conv2dOptions(48, 64, 3).padding(1).bias(false)));
        bn3x3_2 = register_module("bn3x3_2", torch::nn::BatchNorm2d(64));

        // Branch 3: 1x1 conv -> 3x3 conv -> 3x3 conv
        branch3x3dbl_1 = register_module("branch3x3dbl_1", torch::nn::Conv2d(
                                             torch::nn::Conv2dOptions(in_channels, 64, 1).bias(false)));
        bn3x3dbl_1 = register_module("bn3x3dbl_1", torch::nn::BatchNorm2d(64));
        branch3x3dbl_2 = register_module("branch3x3dbl_2", torch::nn::Conv2d(
                                             torch::nn::Conv2dOptions(64, 96, 3).padding(1).bias(false)));
        bn3x3dbl_2 = register_module("bn3x3dbl_2", torch::nn::BatchNorm2d(96));
        branch3x3dbl_3 = register_module("branch3x3dbl_3", torch::nn::Conv2d(
                                             torch::nn::Conv2dOptions(96, 96, 3).padding(1).bias(false)));
        bn3x3dbl_3 = register_module("bn3x3dbl_3", torch::nn::BatchNorm2d(96));

        // Branch 4: Avg pool -> 1x1 conv
        branch_pool = register_module("branch_pool", torch::nn::AvgPool2d(
                                          torch::nn::AvgPool2dOptions(3).stride(1).padding(1)));
        branch_pool_conv = register_module("branch_pool_conv", torch::nn::Conv2d(
                                               torch::nn::Conv2dOptions(
                                                   in_channels, pool_features, 1).bias(false)));
        bn_pool = register_module("bn_pool", torch::nn::BatchNorm2d(pool_features));
    }

    torch::Tensor InceptionAModuleImpl::forward(torch::Tensor x)
    {
        // Branch 1
        auto branch1 = torch::relu(bn1x1->forward(branch1x1->forward(x)));

        // Branch 2
        auto branch2 = torch::relu(bn3x3_1->forward(branch3x3_1->forward(x)));
        branch2 = torch::relu(bn3x3_2->forward(branch3x3_2->forward(branch2)));

        // Branch 3
        auto branch3 = torch::relu(bn3x3dbl_1->forward(branch3x3dbl_1->forward(x)));
        branch3 = torch::relu(bn3x3dbl_2->forward(branch3x3dbl_2->forward(branch3)));
        branch3 = torch::relu(bn3x3dbl_3->forward(branch3x3dbl_3->forward(branch3)));

        // Branch 4
        auto branch4 = branch_pool->forward(x);
        branch4 = torch::relu(bn_pool->forward(branch_pool_conv->forward(branch4)));

        // Concatenate along channel dimension
        return torch::cat({branch1, branch2, branch3, branch4}, 1);
    }


    InceptionBModuleImpl::InceptionBModuleImpl(int in_channels)
    {
        // Branch 1: 3x3 max pool
        branch_pool = register_module("branch_pool", torch::nn::MaxPool2d(
                                          torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));

        // Branch 2: 1x1 conv -> 3x3 conv -> 3x3 conv (stride 2)
        branch3x3_1 = register_module("branch3x3_1", torch::nn::Conv2d(
                                          torch::nn::Conv2dOptions(in_channels, 64, 1).bias(false)));
        bn3x3_1 = register_module("bn3x3_1", torch::nn::BatchNorm2d(64));
        branch3x3_2 = register_module("branch3x3_2", torch::nn::Conv2d(
                                          torch::nn::Conv2dOptions(64, 96, 3).padding(1).bias(false)));
        bn3x3_2 = register_module("bn3x3_2", torch::nn::BatchNorm2d(96));
        branch3x3_3 = register_module("branch3x3_3", torch::nn::Conv2d(
                                          torch::nn::Conv2dOptions(96, 96, 3).stride(2).bias(false)));
        bn3x3_3 = register_module("bn3x3_3", torch::nn::BatchNorm2d(96));
    }

    torch::Tensor InceptionBModuleImpl::forward(torch::Tensor x)
    {
        // Branch 1
        auto branch1 = branch_pool->forward(x);

        // Branch 2
        auto branch2 = torch::relu(bn3x3_1->forward(branch3x3_1->forward(x)));
        branch2 = torch::relu(bn3x3_2->forward(branch3x3_2->forward(branch2)));
        branch2 = torch::relu(bn3x3_3->forward(branch3x3_3->forward(branch2)));

        // Concatenate along channel dimension
        return torch::cat({branch1, branch2}, 1);
    }


    InceptionCModuleImpl::InceptionCModuleImpl(int in_channels, int channels_7x7)
    {
        // Branch 1: 1x1 conv
        branch1x1 = register_module("branch1x1", torch::nn::Conv2d(
                                        torch::nn::Conv2dOptions(in_channels, 192, 1).bias(false)));
        bn1x1 = register_module("bn1x1", torch::nn::BatchNorm2d(192));

        // Branch 2: 1x1 conv -> 1x7 conv -> 7x1 conv
        branch7x7_1 = register_module("branch7x7_1", torch::nn::Conv2d(
                                          torch::nn::Conv2dOptions(in_channels, channels_7x7, 1).bias(false)));
        bn7x7_1 = register_module("bn7x7_1", torch::nn::BatchNorm2d(channels_7x7));
        branch7x7_2 = register_module("branch7x7_2", torch::nn::Conv2d(
                                          torch::nn::Conv2dOptions(channels_7x7, channels_7x7, {1, 7}).padding({
                                              0, 3
                                          }).bias(false)));
        bn7x7_2 = register_module("bn7x7_2", torch::nn::BatchNorm2d(channels_7x7));
        branch7x7_3 = register_module("branch7x7_3", torch::nn::Conv2d(
                                          torch::nn::Conv2dOptions(channels_7x7, 192, {7, 1}).padding({3, 0}).bias(
                                              false)));
        bn7x7_3 = register_module("bn7x7_3", torch::nn::BatchNorm2d(192));

        // Branch 3: 1x1 conv -> 1x7 conv -> 7x1 conv -> 1x7 conv -> 7x1 conv
        branch7x7dbl_1 = register_module("branch7x7dbl_1", torch::nn::Conv2d(
                                             torch::nn::Conv2dOptions(in_channels, channels_7x7, 1).bias(false)));
        bn7x7dbl_1 = register_module("bn7x7dbl_1", torch::nn::BatchNorm2d(channels_7x7));
        branch7x7dbl_2 = register_module("branch7x7dbl_2", torch::nn::Conv2d(
                                             torch::nn::Conv2dOptions(channels_7x7, channels_7x7, {1, 7}).padding({
                                                 0, 3
                                             }).bias(false)));
        bn7x7dbl_2 = register_module("bn7x7dbl_2", torch::nn::BatchNorm2d(channels_7x7));
        branch7x7dbl_3 = register_module("branch7x7dbl_3", torch::nn::Conv2d(
                                             torch::nn::Conv2dOptions(channels_7x7, channels_7x7, {7, 1}).padding({
                                                 3, 0
                                             }).bias(false)));
        bn7x7dbl_3 = register_module("bn7x7dbl_3", torch::nn::BatchNorm2d(channels_7x7));
        branch7x7dbl_4 = register_module("branch7x7dbl_4", torch::nn::Conv2d(
                                             torch::nn::Conv2dOptions(channels_7x7, channels_7x7, {1, 7}).padding({
                                                 0, 3
                                             }).bias(false)));
        bn7x7dbl_4 = register_module("bn7x7dbl_4", torch::nn::BatchNorm2d(channels_7x7));
        branch7x7dbl_5 = register_module("branch7x7dbl_5", torch::nn::Conv2d(
                                             torch::nn::Conv2dOptions(channels_7x7, 192, {7, 1}).padding({3, 0}).
                                             bias(false)));
        bn7x7dbl_5 = register_module("bn7x7dbl_5", torch::nn::BatchNorm2d(192));

        // Branch 4: Avg pool -> 1x1 conv
        branch_pool = register_module("branch_pool", torch::nn::AvgPool2d(
                                          torch::nn::AvgPool2dOptions(3).stride(1).padding(1)));
        branch_pool_conv = register_module("branch_pool_conv", torch::nn::Conv2d(
                                               torch::nn::Conv2dOptions(in_channels, 192, 1).bias(false)));
        bn_pool = register_module("bn_pool", torch::nn::BatchNorm2d(192));
    }

    torch::Tensor InceptionCModuleImpl::forward(torch::Tensor x)
    {
        // Branch 1
        auto branch1 = torch::relu(bn1x1->forward(branch1x1->forward(x)));

        // Branch 2
        auto branch2 = torch::relu(bn7x7_1->forward(branch7x7_1->forward(x)));
        branch2 = torch::relu(bn7x7_2->forward(branch7x7_2->forward(branch2)));
        branch2 = torch::relu(bn7x7_3->forward(branch7x7_3->forward(branch2)));

        // Branch 3
        auto branch3 = torch::relu(bn7x7dbl_1->forward(branch7x7dbl_1->forward(x)));
        branch3 = torch::relu(bn7x7dbl_2->forward(branch7x7dbl_2->forward(branch3)));
        branch3 = torch::relu(bn7x7dbl_3->forward(branch7x7dbl_3->forward(branch3)));
        branch3 = torch::relu(bn7x7dbl_4->forward(branch7x7dbl_4->forward(branch3)));
        branch3 = torch::relu(bn7x7dbl_5->forward(branch7x7dbl_5->forward(branch3)));

        // Branch 4
        auto branch4 = branch_pool->forward(x);
        branch4 = torch::relu(bn_pool->forward(branch_pool_conv->forward(branch4)));

        // Concatenate along channel dimension
        return torch::cat({branch1, branch2, branch3, branch4}, 1);
    }


    AuxClassifierImpl::AuxClassifierImpl(int in_channels, int num_classes)
    {
        conv = register_module("conv", torch::nn::Conv2d(
                                   torch::nn::Conv2dOptions(in_channels, 128, 1).bias(false)));
        bn = register_module("bn", torch::nn::BatchNorm2d(128));
        fc1 = register_module("fc1", torch::nn::Linear(128 * 4 * 4, 1024));
        fc2 = register_module("fc2", torch::nn::Linear(1024, num_classes));
        pool = register_module("pool", torch::nn::AvgPool2d(
                                   torch::nn::AvgPool2dOptions(5).stride(3)));
    }

    torch::Tensor AuxClassifierImpl::forward(torch::Tensor x)
    {
        x = pool->forward(x);
        x = torch::relu(bn->forward(conv->forward(x)));
        x = x.view({x.size(0), -1});
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }


    ReductionAModuleImpl::ReductionAModuleImpl(int in_channels, int k, int l, int m, int n)
    {
        // Branch 1: 3x3 max pool
        branch_pool = register_module("branch_pool", torch::nn::MaxPool2d(
                                          torch::nn::MaxPool2dOptions(3).stride(2)));

        // Branch 2: 3x3 conv (stride 2)
        branch3x3 = register_module("branch3x3", torch::nn::Conv2d(
                                        torch::nn::Conv2dOptions(in_channels, n, 3).stride(2).bias(false)));
        bn3x3 = register_module("bn3x3", torch::nn::BatchNorm2d(n));

        // Branch 3: 1x1 conv -> 3x3 conv -> 3x3 conv (stride 2)
        branch3x3dbl_1 = register_module("branch3x3dbl_1", torch::nn::Conv2d(
                                             torch::nn::Conv2dOptions(in_channels, k, 1).bias(false)));
        bn3x3dbl_1 = register_module("bn3x3dbl_1", torch::nn::BatchNorm2d(k));
        branch3x3dbl_2 = register_module("branch3x3dbl_2", torch::nn::Conv2d(
                                             torch::nn::Conv2dOptions(k, l, 3).padding(1).bias(false)));
        bn3x3dbl_2 = register_module("bn3x3dbl_2", torch::nn::BatchNorm2d(l));
        branch3x3dbl_3 = register_module("branch3x3dbl_3", torch::nn::Conv2d(
                                             torch::nn::Conv2dOptions(l, m, 3).stride(2).bias(false)));
        bn3x3dbl_3 = register_module("bn3x3dbl_3", torch::nn::BatchNorm2d(m));
    }

    torch::Tensor ReductionAModuleImpl::forward(torch::Tensor x)
    {
        auto branch1 = branch_pool->forward(x);
        auto branch2 = torch::relu(bn3x3->forward(branch3x3->forward(x)));
        auto branch3 = torch::relu(bn3x3dbl_1->forward(branch3x3dbl_1->forward(x)));
        branch3 = torch::relu(bn3x3dbl_2->forward(branch3x3dbl_2->forward(branch3)));
        branch3 = torch::relu(bn3x3dbl_3->forward(branch3x3dbl_3->forward(branch3)));
        return torch::cat({branch1, branch2, branch3}, 1);
    }

    ReductionBModuleImpl::ReductionBModuleImpl(int in_channels)
    {
        // Branch 1: 3x3 max pool
        branch_pool = register_module("branch_pool", torch::nn::MaxPool2d(
                                          torch::nn::MaxPool2dOptions(3).stride(2)));

        // Branch 2: 1x1 conv -> 3x3 conv (stride 2)
        branch3x3_1 = register_module("branch3x3_1", torch::nn::Conv2d(
                                          torch::nn::Conv2dOptions(in_channels, 192, 1).bias(false)));
        bn3x3_1 = register_module("bn3x3_1", torch::nn::BatchNorm2d(192));
        branch3x3_2 = register_module("branch3x3_2", torch::nn::Conv2d(
                                          torch::nn::Conv2dOptions(192, 192, 3).stride(2).bias(false)));
        bn3x3_2 = register_module("bn3x3_2", torch::nn::BatchNorm2d(192));

        // Branch 3: 1x1 conv -> 1x7 conv -> 7x1 conv -> 3x3 conv (stride 2)
        branch7x7_1 = register_module("branch7x7_1", torch::nn::Conv2d(
                                          torch::nn::Conv2dOptions(in_channels, 256, 1).bias(false)));
        bn7x7_1 = register_module("bn7x7_1", torch::nn::BatchNorm2d(256));
        branch7x7_2 = register_module("branch7x7_2", torch::nn::Conv2d(
                                          torch::nn::Conv2dOptions(256, 256, {1, 7}).padding({0, 3}).bias(false)));
        bn7x7_2 = register_module("bn7x7_2", torch::nn::BatchNorm2d(256));
        branch7x7_3 = register_module("branch7x7_3", torch::nn::Conv2d(
                                          torch::nn::Conv2dOptions(256, 320, {7, 1}).padding({3, 0}).bias(false)));
        bn7x7_3 = register_module("bn7x7_3", torch::nn::BatchNorm2d(320));
        branch7x7_4 = register_module("branch7x7_4", torch::nn::Conv2d(
                                          torch::nn::Conv2dOptions(320, 320, 3).stride(2).bias(false)));
        bn7x7_4 = register_module("bn7x7_4", torch::nn::BatchNorm2d(320));
    }

    torch::Tensor ReductionBModuleImpl::forward(torch::Tensor x)
    {
        auto branch1 = branch_pool->forward(x);
        auto branch2 = torch::relu(bn3x3_1->forward(branch3x3_1->forward(x)));
        branch2 = torch::relu(bn3x3_2->forward(branch3x3_2->forward(branch2)));
        auto branch3 = torch::relu(bn7x7_1->forward(branch7x7_1->forward(x)));
        branch3 = torch::relu(bn7x7_2->forward(branch7x7_2->forward(branch3)));
        branch3 = torch::relu(bn7x7_3->forward(branch7x7_3->forward(branch3)));
        branch3 = torch::relu(bn7x7_4->forward(branch7x7_4->forward(branch3)));
        return torch::cat({branch1, branch2, branch3}, 1);
    }


    // InceptionV1 (GoogLeNet)
    InceptionV1Impl::InceptionV1Impl(int num_classes)
    {
        // Stem
        conv1 = register_module("conv1", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(3, 64, 7).stride(1).padding(3).bias(false)));
        // Simplified stride
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
        pool1 = register_module("pool1", torch::nn::MaxPool2d(
                                    torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(64, 64, 1).bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(64));
        conv3 = register_module("conv3", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(64, 192, 3).padding(1).bias(false)));
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(192));
        pool2 = register_module("pool2", torch::nn::MaxPool2d(
                                    torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));

        // Inception modules: {in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj}
        inception3a = register_module("inception3a", InceptionModule(192, 64, 96, 128, 16, 32, 32));
        inception3b = register_module("inception3b", InceptionModule(256, 128, 128, 192, 32, 96, 64));
        pool3 = register_module("pool3", torch::nn::MaxPool2d(
                                    torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
        inception4a = register_module("inception4a", InceptionModule(480, 192, 96, 208, 16, 48, 64));
        inception4b = register_module("inception4b", InceptionModule(512, 160, 112, 224, 24, 64, 64));
        inception4c = register_module("inception4c", InceptionModule(512, 128, 128, 256, 24, 64, 64));
        inception4d = register_module("inception4d", InceptionModule(512, 112, 144, 288, 32, 64, 64));
        inception4e = register_module("inception4e", InceptionModule(528, 256, 160, 320, 32, 128, 128));
        pool4 = register_module("pool4", torch::nn::MaxPool2d(
                                    torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
        inception5a = register_module("inception5a", InceptionModule(832, 256, 160, 320, 32, 128, 128));
        inception5b = register_module("inception5b", InceptionModule(832, 384, 192, 384, 48, 128, 128));

        // Head
        avg_pool = register_module("avg_pool", torch::nn::AdaptiveAvgPool2d(
                                       torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
        dropout = register_module("dropout", torch::nn::Dropout(0.4));
        fc = register_module("fc", torch::nn::Linear(1024, num_classes));
    }

    torch::Tensor InceptionV1Impl::forward(torch::Tensor x)
    {
        // Stem: [batch, 3, 32, 32]
        x = torch::relu(bn1->forward(conv1->forward(x))); // [batch, 64, 32, 32]
        x = pool1->forward(x); // [batch, 64, 16, 16]
        x = torch::relu(bn2->forward(conv2->forward(x))); // [batch, 64, 16, 16]
        x = torch::relu(bn3->forward(conv3->forward(x))); // [batch, 192, 16, 16]
        x = pool2->forward(x); // [batch, 192, 8, 8]

        // Inception modules
        x = inception3a->forward(x); // [batch, 256, 8, 8]
        x = inception3b->forward(x); // [batch, 480, 8, 8]
        x = pool3->forward(x); // [batch, 480, 4, 4]
        x = inception4a->forward(x); // [batch, 512, 4, 4]
        x = inception4b->forward(x); // [batch, 512, 4, 4]
        x = inception4c->forward(x); // [batch, 512, 4, 4]
        x = inception4d->forward(x); // [batch, 528, 4, 4]
        x = inception4e->forward(x); // [batch, 832, 4, 4]
        x = pool4->forward(x); // [batch, 832, 2, 2]
        x = inception5a->forward(x); // [batch, 832, 2, 2]
        x = inception5b->forward(x); // [batch, 1024, 2, 2]

        // Head
        x = avg_pool->forward(x); // [batch, 1024, 1, 1]
        x = x.view({x.size(0), -1}); // [batch, 1024]
        x = dropout->forward(x);
        x = fc->forward(x); // [batch, num_classes]
        return x;
    }


    // InceptionV1::InceptionV1(int num_classes, int in_channels)
    // {
    // }
    //
    // InceptionV1::InceptionV1(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    // {
    // }
    //
    // void InceptionV1::reset()
    // {
    // }
    //
    // auto InceptionV1::forward(std::initializer_list<std::any> tensors) -> std::any
    // {
    //     std::vector<std::any> any_vec(tensors);
    //
    //     std::vector<torch::Tensor> tensor_vec;
    //     for (const auto& item : any_vec)
    //     {
    //         tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
    //     }
    //
    //     torch::Tensor x = tensor_vec[0];
    //
    //     return x;
    // }


    InceptionV2Impl::InceptionV2Impl(int num_classes = 10)
    {
        // Stem
        conv1 = register_module("conv1", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(3, 32, 3).stride(1).padding(1).bias(false)));
        // Simplified stride
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));
        conv2 = register_module("conv2", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(32, 32, 3).padding(1).bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(32));
        conv3 = register_module("conv3", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(32, 64, 3).padding(1).bias(false)));
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(64));
        pool1 = register_module("pool1", torch::nn::MaxPool2d(
                                    torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
        conv4 = register_module("conv4", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(64, 80, 1).bias(false)));
        bn4 = register_module("bn4", torch::nn::BatchNorm2d(80));
        conv5 = register_module("conv5", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(80, 192, 3).padding(1).bias(false)));
        bn5 = register_module("bn5", torch::nn::BatchNorm2d(192));
        pool2 = register_module("pool2", torch::nn::MaxPool2d(
                                    torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));

        // Inception modules
        inception3a = register_module("inception3a", InceptionAModule(192, 32));
        inception3b = register_module("inception3b", InceptionAModule(256, 64));
        inception3c = register_module("inception3c", InceptionAModule(320, 64));
        inception4a = register_module("inception4a", InceptionBModule(384));
        inception5a = register_module("inception5a", InceptionAModule(480, 128));
        inception5b = register_module("inception5b", InceptionAModule(608, 128));
        inception5c = register_module("inception5c", InceptionAModule(736, 128));
        inception5d = register_module("inception5d", InceptionAModule(736, 128));
        inception6a = register_module("inception6a", InceptionBModule(736));
        inception7a = register_module("inception7a", InceptionCModule(832, 128));
        inception7b = register_module("inception7b", InceptionCModule(832, 160));
        inception7c = register_module("inception7c", InceptionCModule(832, 192));

        // Head
        avg_pool = register_module("avg_pool", torch::nn::AdaptiveAvgPool2d(
                                       torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
        dropout = register_module("dropout", torch::nn::Dropout(0.4));
        fc = register_module("fc", torch::nn::Linear(768, num_classes));
    }

    torch::Tensor InceptionV2Impl::forward(torch::Tensor x)
    {
        // Stem: [batch, 3, 32, 32]
        x = torch::relu(bn1->forward(conv1->forward(x))); // [batch, 32, 32, 32]
        x = torch::relu(bn2->forward(conv2->forward(x))); // [batch, 32, 32, 32]
        x = torch::relu(bn3->forward(conv3->forward(x))); // [batch, 64, 32, 32]
        x = pool1->forward(x); // [batch, 64, 16, 16]
        x = torch::relu(bn4->forward(conv4->forward(x))); // [batch, 80, 16, 16]
        x = torch::relu(bn5->forward(conv5->forward(x))); // [batch, 192, 16, 16]
        x = pool2->forward(x); // [batch, 192, 8, 8]

        // Inception modules
        x = inception3a->forward(x); // [batch, 256, 8, 8]
        x = inception3b->forward(x); // [batch, 320, 8, 8]
        x = inception3c->forward(x); // [batch, 384, 8, 8]
        x = inception4a->forward(x); // [batch, 480, 4, 4]
        x = inception5a->forward(x); // [batch, 608, 4, 4]
        x = inception5b->forward(x); // [batch, 736, 4, 4]
        x = inception5c->forward(x); // [batch, 736, 4, 4]
        x = inception5d->forward(x); // [batch, 736, 4, 4]
        x = inception6a->forward(x); // [batch, 832, 2, 2]
        x = inception7a->forward(x); // [batch, 832, 2, 2]
        x = inception7b->forward(x); // [batch, 832, 2, 2]
        x = inception7c->forward(x); // [batch, 768, 2, 2]

        // Head
        x = avg_pool->forward(x); // [batch, 768, 1, 1]
        x = x.view({x.size(0), -1}); // [batch, 768]
        x = dropout->forward(x);
        x = fc->forward(x); // [batch, num_classes]
        return x;
    }


    // InceptionV2::InceptionV2(int num_classes, int in_channels)
    // {
    // }
    //
    // InceptionV2::InceptionV2(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    // {
    // }
    //
    // void InceptionV2::reset()
    // {
    // }
    //
    // auto InceptionV2::forward(std::initializer_list<std::any> tensors) -> std::any
    // {
    //     std::vector<std::any> any_vec(tensors);
    //
    //     std::vector<torch::Tensor> tensor_vec;
    //     for (const auto& item : any_vec)
    //     {
    //         tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
    //     }
    //
    //     torch::Tensor x = tensor_vec[0];
    //
    //     return x;
    // }

    InceptionV3Impl::InceptionV3Impl(int num_classes, bool aux_logits) : aux_logits_(aux_logits)
    {
        // Stem
        conv1 = register_module("conv1", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(3, 32, 3).stride(1).padding(1).bias(false)));
        // Simplified stride
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));
        conv2 = register_module("conv2", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(32, 32, 3).padding(1).bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(32));
        conv3 = register_module("conv3", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(32, 64, 3).padding(1).bias(false)));
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(64));
        pool1 = register_module("pool1", torch::nn::MaxPool2d(
                                    torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
        conv4 = register_module("conv4", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(64, 80, 1).bias(false)));
        bn4 = register_module("bn4", torch::nn::BatchNorm2d(80));
        conv5 = register_module("conv5", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(80, 192, 3).padding(1).bias(false)));
        bn5 = register_module("bn5", torch::nn::BatchNorm2d(192));
        pool2 = register_module("pool2", torch::nn::MaxPool2d(
                                    torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));

        // Inception modules
        inception_a1 = register_module("inception_a1", InceptionAModule(192, 32));
        inception_a2 = register_module("inception_a2", InceptionAModule(256, 64));
        inception_a3 = register_module("inception_a3", InceptionAModule(288, 64));
        inception_b = register_module("inception_b", InceptionBModule(288));
        inception_c1 = register_module("inception_c1", InceptionAModule(768, 128));
        inception_c2 = register_module("inception_c2", InceptionAModule(768, 128));
        inception_c3 = register_module("inception_c3", InceptionAModule(768, 128));
        inception_c4 = register_module("inception_c4", InceptionAModule(768, 128));
        inception_d = register_module("inception_d", InceptionBModule(768));
        inception_e1 = register_module("inception_e1", InceptionCModule(1280, 192));
        inception_e2 = register_module("inception_e2", InceptionCModule(2048, 320));

        // Auxiliary classifier
        if (aux_logits_)
        {
            aux_classifier = register_module("aux_classifier", AuxClassifier(768, num_classes));
        }

        // Head
        avg_pool = register_module("avg_pool", torch::nn::AdaptiveAvgPool2d(
                                       torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
        dropout = register_module("dropout", torch::nn::Dropout(0.5));
        fc = register_module("fc", torch::nn::Linear(2048, num_classes));
    }

    std::tuple<torch::Tensor, torch::Tensor> InceptionV3Impl::forward(torch::Tensor x)
    {
        // Stem: [batch, 3, 32, 32]
        x = torch::relu(bn1->forward(conv1->forward(x))); // [batch, 32, 32, 32]
        x = torch::relu(bn2->forward(conv2->forward(x))); // [batch, 32, 32, 32]
        x = torch::relu(bn3->forward(conv3->forward(x))); // [batch, 64, 32, 32]
        x = pool1->forward(x); // [batch, 64, 16, 16]
        x = torch::relu(bn4->forward(conv4->forward(x))); // [batch, 80, 16, 16]
        x = torch::relu(bn5->forward(conv5->forward(x))); // [batch, 192, 16, 16]
        x = pool2->forward(x); // [batch, 192, 8, 8]

        // Inception-A
        x = inception_a1->forward(x); // [batch, 256, 8, 8]
        x = inception_a2->forward(x); // [batch, 288, 8, 8]
        x = inception_a3->forward(x); // [batch, 288, 8, 8]

        // Inception-B
        x = inception_b->forward(x); // [batch, 768, 4, 4]

        // Inception-C
        x = inception_c1->forward(x); // [batch, 768, 4, 4]
        x = inception_c2->forward(x); // [batch, 768, 4, 4]
        x = inception_c3->forward(x); // [batch, 768, 4, 4]
        torch::Tensor aux_output;
        if (aux_logits_ && training())
        {
            aux_output = aux_classifier->forward(x);
        }
        x = inception_c4->forward(x); // [batch, 768, 4, 4]

        // Inception-D
        x = inception_d->forward(x); // [batch, 1280, 2, 2]

        // Inception-E
        x = inception_e1->forward(x); // [batch, 2048, 2, 2]
        x = inception_e2->forward(x); // [batch, 2048, 2, 2]

        // Head
        x = avg_pool->forward(x); // [batch, 2048, 1, 1]
        x = x.view({x.size(0), -1}); // [batch, 2048]
        x = dropout->forward(x);
        x = fc->forward(x); // [batch, num_classes]

        return std::make_tuple(x, aux_output);
    }


    // InceptionV3::InceptionV3(int num_classes, int in_channels)
    // {
    // }
    //
    // InceptionV3::InceptionV3(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    // {
    // }
    //
    // void InceptionV3::reset()
    // {
    // }
    //
    // auto InceptionV3::forward(std::initializer_list<std::any> tensors) -> std::any
    // {
    //     std::vector<std::any> any_vec(tensors);
    //
    //     std::vector<torch::Tensor> tensor_vec;
    //     for (const auto& item : any_vec)
    //     {
    //         tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
    //     }
    //
    //     torch::Tensor x = tensor_vec[0];
    //
    //     return x;
    // }

    InceptionV4Impl::InceptionV4Impl(int num_classes)
    {
        // Stem
        conv1 = register_module("conv1", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(3, 32, 3).stride(1).padding(1).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));
        conv2 = register_module("conv2", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(32, 32, 3).padding(1).bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(32));
        conv3 = register_module("conv3", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(32, 64, 3).padding(1).bias(false)));
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(64));
        pool1 = register_module("pool1", torch::nn::MaxPool2d(
                                    torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
        conv4 = register_module("conv4", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(64, 96, 3).padding(1).bias(false)));
        bn4 = register_module("bn4", torch::nn::BatchNorm2d(96));
        conv5 = register_module("conv5", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(96, 64, 1).bias(false)));
        bn5 = register_module("bn5", torch::nn::BatchNorm2d(64));
        conv6 = register_module("conv6", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(64, 96, 3).padding(1).bias(false)));
        bn6 = register_module("bn6", torch::nn::BatchNorm2d(96));
        conv7 = register_module("conv7", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(160, 192, 3).stride(2).bias(false)));
        bn7 = register_module("bn7", torch::nn::BatchNorm2d(192));

        // Inception modules
        inception_a1 = register_module("inception_a1", InceptionAModule(192));
        inception_a2 = register_module("inception_a2", InceptionAModule(384));
        inception_a3 = register_module("inception_a3", InceptionAModule(384));
        inception_a4 = register_module("inception_a4", InceptionAModule(384));
        reduction_a = register_module("reduction_a", ReductionAModule(384, 192, 224, 256, 384));
        inception_b1 = register_module("inception_b1", InceptionBModule(1024));
        inception_b2 = register_module("inception_b2", InceptionBModule(1024));
        inception_b3 = register_module("inception_b3", InceptionBModule(1024));
        inception_b4 = register_module("inception_b4", InceptionBModule(1024));
        inception_b5 = register_module("inception_b5", InceptionBModule(1024));
        inception_b6 = register_module("inception_b6", InceptionBModule(1024));
        inception_b7 = register_module("inception_b7", InceptionBModule(1024));
        reduction_b = register_module("reduction_b", ReductionBModule(1024));
        inception_c1 = register_module("inception_c1", InceptionCModule(1536));
        inception_c2 = register_module("inception_c2", InceptionCModule(1536));
        inception_c3 = register_module("inception_c3", InceptionCModule(1536));

        // Head
        avg_pool = register_module("avg_pool", torch::nn::AdaptiveAvgPool2d(
                                       torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
        dropout = register_module("dropout", torch::nn::Dropout(0.8));
        fc = register_module("fc", torch::nn::Linear(1536, num_classes));
    }

    torch::Tensor InceptionV4Impl::forward(torch::Tensor x)
    {
        // Stem: [batch, 3, 32, 32]
        x = torch::relu(bn1->forward(conv1->forward(x))); // [batch, 32, 32, 32]
        x = torch::relu(bn2->forward(conv2->forward(x))); // [batch, 32, 32, 32]
        x = torch::relu(bn3->forward(conv3->forward(x))); // [batch, 64, 32, 32]
        x = pool1->forward(x); // [batch, 64, 16, 16]
        x = torch::relu(bn4->forward(conv4->forward(x))); // [batch, 96, 16, 16]
        auto branch1 = torch::relu(bn5->forward(conv5->forward(x))); // [batch, 64, 16, 16]
        auto branch2 = torch::relu(bn6->forward(conv6->forward(x))); // [batch, 96, 16, 16]
        x = torch::cat({branch1, branch2}, 1); // [batch, 160, 16, 16]
        x = torch::relu(bn7->forward(conv7->forward(x))); // [batch, 192, 8, 8]

        // Inception-A
        x = inception_a1->forward(x); // [batch, 384, 8, 8]
        x = inception_a2->forward(x); // [batch, 384, 8, 8]
        x = inception_a3->forward(x); // [batch, 384, 8, 8]
        x = inception_a4->forward(x); // [batch, 384, 8, 8]

        // Reduction-A
        x = reduction_a->forward(x); // [batch, 1024, 4, 4]

        // Inception-B
        x = inception_b1->forward(x); // [batch, 1024, 4, 4]
        x = inception_b2->forward(x); // [batch, 1024, 4, 4]
        x = inception_b3->forward(x); // [batch, 1024, 4, 4]
        x = inception_b4->forward(x); // [batch, 1024, 4, 4]
        x = inception_b5->forward(x); // [batch, 1024, 4, 4]
        x = inception_b6->forward(x); // [batch, 1024, 4, 4]
        x = inception_b7->forward(x); // [batch, 1024, 4, 4]

        // Reduction-B
        x = reduction_b->forward(x); // [batch, 1536, 2, 2]

        // Inception-C
        x = inception_c1->forward(x); // [batch, 1536, 2, 2]
        x = inception_c2->forward(x); // [batch, 1536, 2, 2]
        x = inception_c3->forward(x); // [batch, 1536, 2, 2]

        // Head
        x = avg_pool->forward(x); // [batch, 1536, 1, 1]
        x = x.view({x.size(0), -1}); // [batch, 1536]
        x = dropout->forward(x);
        x = fc->forward(x); // [batch, num_classes]
        return x;
    }


    // InceptionV4::InceptionV4(int num_classes, int in_channels)
    // {
    // }
    //
    // InceptionV4::InceptionV4(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    // {
    // }
    //
    // void InceptionV4::reset()
    // {
    // }
    //
    // auto InceptionV4::forward(std::initializer_list<std::any> tensors) -> std::any
    // {
    //     std::vector<std::any> any_vec(tensors);
    //
    //     std::vector<torch::Tensor> tensor_vec;
    //     for (const auto& item : any_vec)
    //     {
    //         tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
    //     }
    //
    //     torch::Tensor x = tensor_vec[0];
    //
    //     return x;
    // }
}
