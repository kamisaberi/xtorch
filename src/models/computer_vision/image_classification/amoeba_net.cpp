#include <models/computer_vision/image_classification/amoeba_net.h>


using namespace std;
//AmoabaNet GROK

// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // Operation: 3x3 Convolution
// struct Conv3x3Impl : torch::nn::Module {
//     Conv3x3Impl(int in_channels, int out_channels) {
//         conv = register_module("conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(1).padding(1)));
//         bn = register_module("bn", torch::nn::BatchNorm2d(out_channels));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         return torch::relu(bn->forward(conv->forward(x)));
//     }
//
//     torch::nn::Conv2d conv{nullptr};
//     torch::nn::BatchNorm2d bn{nullptr};
// };
// TORCH_MODULE(Conv3x3);
//
// // Operation: 1x1 Convolution
// struct Conv1x1Impl : torch::nn::Module {
//     Conv1x1Impl(int in_channels, int out_channels) {
//         conv = register_module("conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(1)));
//         bn = register_module("bn", torch::nn::BatchNorm2d(out_channels));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         return torch::relu(bn->forward(conv->forward(x)));
//     }
//
//     torch::nn::Conv2d conv{nullptr};
//     torch::nn::BatchNorm2d bn{nullptr};
// };
// TORCH_MODULE(Conv1x1);
//
// // Operation: 3x3 Max Pool
// struct MaxPool3x3Impl : torch::nn::Module {
//     MaxPool3x3Impl() {
//         pool = register_module("pool", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(3).stride(1).padding(1)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         return pool->forward(x);
//     }
//
//     torch::nn::MaxPool2d pool{nullptr};
// };
// TORCH_MODULE(MaxPool3x3);
//
// // Normal Cell
// struct NormalCellImpl : torch::nn::Module {
//     NormalCellImpl(int prev_channels, int channels) {
//         // Simplified: Two branches (Conv3x3 + MaxPool3x3, Conv1x1)
//         op1 = register_module("op1", Conv3x3(prev_channels, channels));
//         op2 = register_module("op2", MaxPool3x3());
//         op3 = register_module("op3", Conv1x1(prev_channels, channels));
//     }
//
//     torch::Tensor forward(torch::Tensor prev, torch::Tensor curr) {
//         // Branch 1: Conv3x3(prev) + MaxPool3x3(curr)
//         auto b1 = op1->forward(prev) + op2->forward(curr);
//         // Branch 2: Conv1x1(curr)
//         auto b2 = op3->forward(curr);
//         // Combine
//         return torch::cat({b1, b2}, 1); // [batch, 2*channels, h, w]
//     }
//
//     Conv3x3 op1{nullptr};
//     MaxPool3x3 op2{nullptr};
//     Conv1x1 op3{nullptr};
// };
// TORCH_MODULE(NormalCell);
//
// // Reduction Cell
// struct ReductionCellImpl : torch::nn::Module {
//     ReductionCellImpl(int prev_channels, int channels) {
//         // Simplified: Two branches (Conv3x3 stride 2, MaxPool3x3 stride 2)
//         op1 = register_module("op1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(prev_channels, channels, 3).stride(2).padding(1)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(channels));
//         op2 = register_module("op2", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
//         op3 = register_module("op3", Conv1x1(prev_channels, channels));
//     }
//
//     torch::Tensor forward(torch::Tensor prev, torch::Tensor curr) {
//         // Branch 1: Conv3x3 stride 2(prev) + MaxPool3x3 stride 2(curr)
//         auto b1 = torch::relu(bn1->forward(op1->forward(prev))) + op2->forward(curr);
//         // Branch 2: Conv1x1(curr)
//         auto b2 = op3->forward(curr);
//         // Combine
//         return torch::cat({b1, b2}, 1); // [batch, 2*channels, h/2, w/2]
//     }
//
//     torch::nn::Conv2d op1{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr};
//     torch::nn::MaxPool2d op2{nullptr};
//     Conv1x1 op3{nullptr};
// };
// TORCH_MODULE(ReductionCell);
//
// // AmoebaNet-A (Simplified)
// struct AmoebaNetImpl : torch::nn::Module {
//     AmoebaNetImpl(int in_channels, int num_classes, int channels = 64) {
//         stem = register_module("stem", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, channels, 3).stride(1).padding(1)));
//         bn_stem = register_module("bn_stem", torch::nn::BatchNorm2d(channels));
//         normal_cell = register_module("normal_cell", NormalCell(channels, channels));
//         reduction_cell = register_module("reduction_cell", ReductionCell(channels * 2, channels));
//         classifier = register_module("classifier", torch::nn::Linear(4 * channels, num_classes));
//         pool = register_module("pool", torch::nn::AdaptiveAvgPool2d(1));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, in_channels, 32, 32]
//         auto h = torch::relu(bn_stem->forward(stem->forward(x))); // [batch, channels, 32, 32]
//         auto prev = h;
//         // Normal cell
//         h = normal_cell->forward(prev, h); // [batch, 2*channels, 32, 32]
//         prev = h;
//         // Reduction cell
//         h = reduction_cell->forward(prev, h); // [batch, 4*channels, 16, 16]
//         // Global average pooling
//         h = pool->forward(h); // [batch, 4*channels, 1, 1]
//         h = h.view({h.size(0), -1}); // [batch, 4*channels]
//         // Classifier
//         return classifier->forward(h); // [batch, num_classes]
//     }
//
//     torch::nn::Conv2d stem{nullptr};
//     torch::nn::BatchNorm2d bn_stem{nullptr};
//     NormalCell normal_cell{nullptr};
//     ReductionCell reduction_cell{nullptr};
//     torch::nn::Linear classifier{nullptr};
//     torch::nn::AdaptiveAvgPool2d pool{nullptr};
// };
// TORCH_MODULE(AmoebaNet);
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
//         const int channels = 64;
//         const int batch_size = 32;
//         const float lr = 0.001;
//         const int num_epochs = 10;
//
//         // Initialize model
//         AmoebaNet model(in_channels, num_classes, channels);
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
//                 torch::save(model, "amoebanet_epoch_" + std::to_string(epoch + 1) + ".pt");
//             }
//         }
//
//         // Save final model
//         torch::save(model, "amoebanet.pt");
//         std::cout << "Model saved as amoebanet.pt" << std::endl;
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
    Conv3x3::Conv3x3(int in_channels, int out_channels)
    {
        conv = register_module("conv", torch::nn::Conv2d(
                                   torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(1).padding(1)));
        bn = register_module("bn", torch::nn::BatchNorm2d(out_channels));
    }

    auto Conv3x3::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
        x = x.to(torch::kFloat32);
        return this->forward(x);
    }

    torch::Tensor Conv3x3::forward(torch::Tensor x)
    {
        return torch::relu(bn->forward(conv->forward(x)));
    }

    Conv1x1::Conv1x1(int in_channels, int out_channels)
    {
        conv = register_module("conv", torch::nn::Conv2d(
                                   torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(1)));
        bn = register_module("bn", torch::nn::BatchNorm2d(out_channels));
    }

    auto Conv1x1::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
        return this->forward(x);
    }

    torch::Tensor Conv1x1::forward(torch::Tensor x)
    {
        return torch::relu(bn->forward(conv->forward(x)));
    }


    MaxPool3x3::MaxPool3x3()
    {
        pool = register_module("pool", torch::nn::MaxPool2d(
                                   torch::nn::MaxPool2dOptions(3).stride(1).padding(1)));
    }

    auto MaxPool3x3::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
        return this->forward(x);
    }

    torch::Tensor MaxPool3x3::forward(torch::Tensor x)
    {
        return pool->forward(x);
    }

    NormalCell::NormalCell(int prev_channels, int channels)
    {
        // Simplified: Two branches (Conv3x3 + MaxPool3x3, Conv1x1)
        op1 = register_module("op1", std::make_shared<Conv3x3>(prev_channels, channels));
        op2 = register_module("op2", std::make_shared<MaxPool3x3>());
        op3 = register_module("op3", std::make_shared<Conv1x1>(prev_channels, channels));
    }

    auto NormalCell::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor prev = tensor_vec[0];
        torch::Tensor curr = tensor_vec[1];
        return this->forward(prev, curr);
    }

    torch::Tensor NormalCell::forward(torch::Tensor prev, torch::Tensor curr)
    {
        // Branch 1: Conv3x3(prev) + MaxPool3x3(curr)
        auto b1 = op1->forward(prev) + op2->forward(curr);
        // Branch 2: Conv1x1(curr)
        auto b2 = op3->forward(curr);
        // Combine
        return torch::cat({b1, b2}, 1); // [batch, 2*channels, h, w]
    }

    ReductionCell::ReductionCell(int prev_channels, int channels)
    {
        // Simplified: Two branches (Conv3x3 stride 2, MaxPool3x3 stride 2)
        op1 = register_module("op1", torch::nn::Conv2d(
                                  torch::nn::Conv2dOptions(prev_channels, channels, 3).stride(2).padding(1)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(channels));
        op2 = register_module("op2", torch::nn::MaxPool2d(
                                  torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
        op3 = register_module("op3", std::make_shared<Conv1x1>(prev_channels, channels));
    }


    auto ReductionCell::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor prev = tensor_vec[0];
        torch::Tensor curr = tensor_vec[1];
        return this->forward(prev, curr);
    }


    torch::Tensor ReductionCell::forward(torch::Tensor prev, torch::Tensor curr)
    {
        // Branch 1: Conv3x3 stride 2(prev) + MaxPool3x3 stride 2(curr)
        auto b1 = torch::relu(bn1->forward(op1->forward(prev))) + op2->forward(curr);
        // Branch 2: Conv1x1(curr)
        auto b2 = op3->forward(curr);
        // Combine
        return torch::cat({b1, b2}, 1); // [batch, 2*channels, h/2, w/2]
    }


    AmoebaNet::AmoebaNet(int in_channels, int num_classes, int channels)
    {
        stem = register_module("stem", torch::nn::Conv2d(
                                   torch::nn::Conv2dOptions(in_channels, channels, 3).stride(1).padding(1)));
        bn_stem = register_module("bn_stem", torch::nn::BatchNorm2d(channels));
        normal_cell = register_module("normal_cell", std::make_shared<NormalCell>(channels, channels));
        reduction_cell = register_module("reduction_cell", std::make_shared<ReductionCell>(channels * 2, channels));
        classifier = register_module("classifier", torch::nn::Linear(4 * channels, num_classes));
        pool = register_module("pool", torch::nn::AdaptiveAvgPool2d(1));
    }

    auto AmoebaNet::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
        return this->forward(x);
    }

    torch::Tensor AmoebaNet::forward(torch::Tensor x)
    {
        // x: [batch, in_channels, 32, 32]
        auto h = torch::relu(bn_stem->forward(stem->forward(x))); // [batch, channels, 32, 32]
        auto prev = h;
        // Normal cell
        h = normal_cell->forward(prev, h); // [batch, 2*channels, 32, 32]
        prev = h;
        // Reduction cell
        h = reduction_cell->forward(prev, h); // [batch, 4*channels, 16, 16]
        // Global average pooling
        h = pool->forward(h); // [batch, 4*channels, 1, 1]
        h = h.view({h.size(0), -1}); // [batch, 4*channels]
        // Classifier
        return classifier->forward(h); // [batch, num_classes]
    }


    // AmoabaNet::AmoabaNet(int num_classes, int in_channels)
    // {
    // }
    //
    // AmoabaNet::AmoabaNet(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    // {
    // }
    //
    // void AmoabaNet::reset()
    // {
    // }
    //
    // auto AmoabaNet::forward(std::initializer_list<std::any> tensors) -> std::any
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
