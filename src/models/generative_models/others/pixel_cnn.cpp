#include "include/models/generative_models/others/pixel_cnn.h"


using namespace std;
//PIXELCNN GROK

// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // Masked Convolution Layer
// struct MaskedConv2dImpl : torch::nn::Module {
//     MaskedConv2dImpl(int in_channels, int out_channels, int kernel_size, bool mask_type_A)
//         : mask_type_A_(mask_type_A) {
//         conv = register_module("conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).stride(1).padding(kernel_size / 2)));
//
//         // Initialize mask
//         auto mask = torch::ones({out_channels, in_channels, kernel_size, kernel_size});
//         int center = kernel_size / 2;
//         if (mask_type_A_) {
//             // Type A: Mask includes center pixel for first layer
//             mask.slice(2, center + 1, kernel_size) = 0; // Mask future rows
//             mask.slice(3, center + 1, kernel_size) = 0; // Mask future columns in center row
//         } else {
//             // Type B: Mask excludes center pixel
//             mask.slice(2, center + 1, kernel_size) = 0; // Mask future rows
//             mask.slice(2, center, center + 1).slice(3, center, kernel_size) = 0; // Mask center and future columns
//         }
//         mask_ = register_buffer("mask", mask);
//
//         // Initialize weights
//         torch::nn::init::kaiming_normal_(conv->weight);
//         if (conv->bias.defined()) {
//             torch::nn::init::zeros_(conv->bias);
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         conv->weight.mul_(mask_);
//         return conv->forward(x);
//     }
//
//     bool mask_type_A_;
//     torch::nn::Conv2d conv{nullptr};
//     torch::Tensor mask_;
// };
// TORCH_MODULE(MaskedConv2d);
//
// // PixelCNN Model
// struct PixelCNNImpl : torch::nn::Module {
//     PixelCNNImpl(int num_levels, int num_filters = 64, int num_layers = 5) : num_levels_(num_levels) {
//         // Initial Type-A masked conv
//         conv1 = register_module("conv1", MaskedConv2d(1, num_filters, 7, true));
//
//         // Residual blocks with Type-B masked convs
//         for (int i = 0; i < num_layers; ++i) {
//             layers->push_back("conv_" + std::to_string(i), MaskedConv2d(num_filters, num_filters, 3, false));
//             layers->push_back("bn_" + std::to_string(i), torch::nn::BatchNorm2d(num_filters));
//             layers->push_back("relu_" + std::to_string(i), torch::nn::ReLU());
//         }
//         layers = register_module("layers", layers);
//
//         // Output conv to predict logits for num_levels
//         out_conv = register_module("out_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(num_filters, num_levels, 1).stride(1)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = relu->forward(conv1->forward(x)); // [batch, num_filters, 28, 28]
//
//         for (int i = 0; i < layers->size() / 3; ++i) {
//             auto residual = x;
//             x = layers[i * 3]->forward(x); // MaskedConv2d
//             x = layers[i * 3 + 1]->forward(x); // BatchNorm2d
//             x = layers[i * 3 + 2]->forward(x); // ReLU
//             x = x + residual; // Residual connection
//         }
//
//         x = out_conv->forward(x); // [batch, num_levels, 28, 28]
//         return x;
//     }
//
//     int num_levels_;
//     MaskedConv2d conv1{nullptr};
//     torch::nn::Sequential layers{nullptr};
//     torch::nn::Conv2d out_conv{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(PixelCNN);
//
// // Custom Dataset for Quantized Grayscale Images
// struct QuantizedImageDataset : torch::data::Dataset<QuantizedImageDataset> {
//     QuantizedImageDataset(const std::string& img_dir, int num_levels) : num_levels_(num_levels) {
//         for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
//             if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
//                 image_paths_.push_back(entry.path().string());
//             }
//         }
//     }
//
//     torch::data::Example<> get(size_t index) override {
//         cv::Mat image = cv::imread(image_paths_[index % image_paths_.size()], cv::IMREAD_GRAYSCALE);
//         if (image.empty()) {
//             throw std::runtime_error("Failed to load image: " + image_paths_[index % image_paths_.size()]);
//         }
//         image.convertTo(image, CV_32F, 1.0 / 255.0); // Normalize to [0, 1]
//
//         // Quantize to num_levels
//         cv::Mat quantized;
//         image.convertTo(quantized, CV_32F, num_levels_ - 1);
//         quantized = quantized.round();
//         torch::Tensor img_tensor = torch::from_blob(quantized.data, {1, image.rows, image.cols}, torch::kFloat32);
//         torch::Tensor label_tensor = torch::from_blob(quantized.data, {image.rows, image.cols}, torch::kInt64);
//
//         return {img_tensor, label_tensor};
//     }
//
//     torch::optional<size_t> size() const override {
//         return image_paths_.size();
//     }
//
//     std::vector<std::string> image_paths_;
//     int num_levels_;
// };
//
// // Generate samples from PixelCNN
// torch::Tensor generate_samples(PixelCNN& model, int num_samples, int height, int width, int num_levels, torch::Device device) {
//     torch::NoGradGuard no_grad;
//     auto samples = torch::zeros({num_samples, 1, height, width}, torch::kFloat32, device);
//
//     for (int i = 0; i < height; ++i) {
//         for (int j = 0; j < width; ++j) {
//             auto logits = model->forward(samples); // [num_samples, num_levels, height, width]
//             auto probs = torch::softmax(logits.slice(2, i, i + 1).slice(3, j, j + 1), 1);
//             probs = probs.squeeze(3).squeeze(2); // [num_samples, num_levels]
//             auto pixel_values = torch::multinomial(probs, 1).to(torch::kFloat32); // [num_samples, 1]
//             samples.index_put_({torch::indexing::Slice(), 0, i, j}, pixel_values / (num_levels - 1));
//         }
//     }
//
//     return samples;
// }
//
// int main() {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Hyperparameters
//         const int num_levels = 4; // Quantized pixel levels (e.g., 0, 1/3, 2/3, 1)
//         const int batch_size = 32;
//         const int num_filters = 64;
//         const int num_layers = 5;
//         const float lr = 0.001;
//         const int num_epochs = 20;
//
//         // Initialize model
//         PixelCNN model(num_levels, num_filters, num_layers);
//         model->to(device);
//
//         // Optimizer
//         torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));
//
//         // Loss function
//         auto ce_loss = torch::nn::CrossEntropyLoss();
//
//         // Load dataset
//         auto dataset = QuantizedImageDataset("./data/images", num_levels)
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
//                 auto labels = batch.target.to(device); // [batch, 28, 28]
//
//                 optimizer.zero_grad();
//                 auto logits = model->forward(images); // [batch, num_levels, 28, 28]
//                 // Reshape for cross-entropy: [batch * 28 * 28, num_levels]
//                 auto loss = ce_loss->forward(logits.permute({0, 2, 3, 1}).reshape({-1, num_levels}), labels.reshape(-1));
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
//             // Generate samples every 5 epochs
//             if ((epoch + 1) % 5 == 0) {
//                 auto samples = generate_samples(model, 1, 28, 28, num_levels, device).squeeze().to(torch::kCPU);
//                 cv::Mat img(28, 28, CV_32F, samples.data_ptr<float>());
//                 img.convertTo(img, CV_8U, 255.0);
//                 cv::imwrite("generated_pixelcnn_epoch_" + std::to_string(epoch + 1) + ".jpg", img);
//             }
//         }
//
//         // Save model
//         torch::save(model, "pixelcnn.pt");
//         std::cout << "Model saved as pixelcnn.pt" << std::endl;
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
    PixelCNN::PixelCNN(int num_classes, int in_channels)
    {
    }

    PixelCNN::PixelCNN(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void PixelCNN::reset()
    {
    }

    auto PixelCNN::forward(std::initializer_list<std::any> tensors) -> std::any
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
