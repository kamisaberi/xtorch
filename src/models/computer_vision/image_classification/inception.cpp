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






namespace xt::models
{
    InceptionV1::InceptionV1(int num_classes, int in_channels)
    {
    }

    InceptionV1::InceptionV1(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void InceptionV1::reset()
    {
    }

    auto InceptionV1::forward(std::initializer_list<std::any> tensors) -> std::any
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





    InceptionV2::InceptionV2(int num_classes, int in_channels)
    {
    }

    InceptionV2::InceptionV2(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void InceptionV2::reset()
    {
    }

    auto InceptionV2::forward(std::initializer_list<std::any> tensors) -> std::any
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


    InceptionV3::InceptionV3(int num_classes, int in_channels)
    {
    }

    InceptionV3::InceptionV3(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void InceptionV3::reset()
    {
    }

    auto InceptionV3::forward(std::initializer_list<std::any> tensors) -> std::any
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


    InceptionV4::InceptionV4(int num_classes, int in_channels)
    {
    }

    InceptionV4::InceptionV4(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void InceptionV4::reset()
    {
    }

    auto InceptionV4::forward(std::initializer_list<std::any> tensors) -> std::any
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
