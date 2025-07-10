#include "include/models/computer_vision/image_segmentation/unet.h"


//UNet GROK

// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // Double Convolution Block
// struct DoubleConvImpl : torch::nn::Module {
//     DoubleConvImpl(int in_channels, int out_channels) {
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, in_channels, h, w]
//         x = torch::relu(bn1->forward(conv1->forward(x))); // [batch, out_channels, h, w]
//         x = torch::relu(bn2->forward(conv2->forward(x))); // [batch, out_channels, h, w]
//         return x;
//     }
//
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
// };
// TORCH_MODULE(DoubleConv);
//
// // U-Net Model
// struct UNetImpl : torch::nn::Module {
//     UNetImpl(int in_channels, int out_channels) {
//         // Encoder
//         enc1 = register_module("enc1", DoubleConv(in_channels, 64));
//         enc2 = register_module("enc2", DoubleConv(64, 128));
//         enc3 = register_module("enc3", DoubleConv(128, 256));
//
//         // Bottleneck
//         bottleneck = register_module("bottleneck", DoubleConv(256, 512));
//
//         // Decoder
//         upconv3 = register_module("upconv3", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(512, 256, 2).stride(2)));
//         dec3 = register_module("dec3", DoubleConv(512, 256));
//         upconv2 = register_module("upconv2", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(256, 128, 2).stride(2)));
//         dec2 = register_module("dec2", DoubleConv(256, 128));
//         upconv1 = register_module("upconv1", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(128, 64, 2).stride(2)));
//         dec1 = register_module("dec1", DoubleConv(128, 64));
//
//         // Output layer
//         out_conv = register_module("out_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, out_channels, 1)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, in_channels, h, w]
//
//         // Encoder
//         auto e1 = enc1->forward(x); // [batch, 64, 28, 28]
//         auto p1 = torch::max_pool2d(e1, 2); // [batch, 64, 14, 14]
//         auto e2 = enc2->forward(p1); // [batch, 128, 14, 14]
//         auto p2 = torch::max_pool2d(e2, 2); // [batch, 128, 7, 7]
//         auto e3 = enc3->forward(p2); // [batch, 256, 7, 7]
//         auto p3 = torch::max_pool2d(e3, 2); // [batch, 256, 3, 3]
//
//         // Bottleneck
//         auto b = bottleneck->forward(p3); // [batch, 512, 3, 3]
//
//         // Decoder
//         auto u3 = upconv3->forward(b); // [batch, 256, 6, 6]
//         // Pad to match e3 size (7x7)
//         u3 = torch::pad(u3, {0, 1, 0, 1}); // [batch, 256, 7, 7]
//         auto d3 = dec3->forward(torch::cat({u3, e3}, 1)); // [batch, 256, 7, 7]
//         auto u2 = upconv2->forward(d3); // [batch, 128, 14, 14]
//         auto d2 = dec2->forward(torch::cat({u2, e2}, 1)); // [batch, 128, 14, 14]
//         auto u1 = upconv1->forward(d2); // [batch, 64, 28, 28]
//         auto d1 = dec1->forward(torch::cat({u1, e1}, 1)); // [batch, 64, 28, 28]
//
//         // Output
//         auto out = out_conv->forward(d1); // [batch, out_channels, 28, 28]
//         return out;
//     }
//
//     DoubleConv enc1{nullptr}, enc2{nullptr}, enc3{nullptr}, bottleneck{nullptr};
//     DoubleConv dec1{nullptr}, dec2{nullptr}, dec3{nullptr};
//     torch::nn::ConvTranspose2d upconv1{nullptr}, upconv2{nullptr}, upconv3{nullptr};
//     torch::nn::Conv2d out_conv{nullptr};
// };
// TORCH_MODULE(UNet);
//
// // Dice Loss for Binary Segmentation
// struct DiceLossImpl : torch::nn::Module {
//     DiceLossImpl(float smooth = 1.0) : smooth_(smooth) {}
//
//     torch::Tensor forward(torch::Tensor input, torch::Tensor target) {
//         // input: [batch, 1, h, w], target: [batch, 1, h, w]
//         input = torch::sigmoid(input); // Convert logits to probabilities
//         auto intersection = (input * target).sum({2, 3}); // [batch, 1]
//         auto union = input.sum({2, 3}) + target.sum({2, 3}); // [batch, 1]
//         auto dice = (2.0 * intersection + smooth_) / (union + smooth_); // [batch, 1]
//         return 1.0 - dice.mean(); // Average loss
//     }
//
//     float smooth_;
// };
// TORCH_MODULE(DiceLoss);
//
// // Dataset for Images and Masks
// struct SegmentationDataset : torch::data::Dataset<SegmentationDataset> {
//     SegmentationDataset(const std::string& img_dir, const std::string& mask_dir) {
//         for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
//             if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
//                 image_paths_.push_back(entry.path().string());
//                 std::string mask_path = mask_dir + "/" + entry.path().filename().string();
//                 mask_paths_.push_back(mask_path);
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
//         image.convertTo(image, CV_32F, 2.0 / 255.0, -1.0); // Normalize to [-1, 1]
//         torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols}, torch::kFloat32);
//
//         // Load mask
//         cv::Mat mask = cv::imread(mask_paths_[index % mask_paths_.size()], cv::IMREAD_GRAYSCALE);
//         if (mask.empty()) {
//             throw std::runtime_error("Failed to load mask: " + mask_paths_[index % mask_paths_.size()]);
//         }
//         mask.convertTo(mask, CV_32F, 1.0 / 255.0); // Normalize to [0, 1]
//         torch::Tensor mask_tensor = torch::from_blob(mask.data, {1, mask.rows, mask.cols}, torch::kFloat32);
//
//         return {img_tensor, mask_tensor};
//     }
//
//     torch::optional<size_t> size() const override {
//         return image_paths_.size();
//     }
//
//     std::vector<std::string> image_paths_, mask_paths_;
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
//         const int out_channels = 1; // Binary segmentation
//         const int batch_size = 16;
//         const float lr = 0.001;
//         const int num_epochs = 20;
//
//         // Initialize model
//         UNet model(in_channels, out_channels);
//         model->to(device);
//
//         // Optimizer
//         torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));
//
//         // Loss function
//         DiceLoss dice_loss;
//
//         // Load dataset
//         auto dataset = SegmentationDataset("./data/images", "./data/masks")
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
//                 auto masks = batch.target.to(device);
//
//                 optimizer.zero_grad();
//                 auto logits = model->forward(images); // [batch, 1, 28, 28]
//                 auto loss = dice_loss.forward(logits, masks);
//                 loss.backward();
//                 optimizer.step();
//
//                 loss_avg += loss.item<float>();
//                 batch_count++;
//             }
//
//             std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "] "
//                       << "Dice Loss: " << loss_avg / batch_count << std::endl;
//
//             // Save model every 10 epochs
//             if ((epoch + 1) % 10 == 0) {
//                 torch::save(model, "unet_epoch_" + std::to_string(epoch + 1) + ".pt");
//             }
//         }
//
//         // Save final model
//         torch::save(model, "unet.pt");
//         std::cout << "Model saved as unet.pt" << std::endl;
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
    DoubleConv::DoubleConv(int in_channels, int out_channels)
    {
        conv1 = register_module("conv1", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));
        conv2 = register_module("conv2", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
    }

    auto DoubleConv::forward(std::initializer_list<std::any> tensors) -> std::any
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

    torch::Tensor DoubleConv::forward(torch::Tensor x)
    {
        x = torch::relu(bn1->forward(conv1->forward(x))); // [batch, out_channels, h, w]
        x = torch::relu(bn2->forward(conv2->forward(x))); // [batch, out_channels, h, w]
        return x;
    }

    UNet::UNet(int in_channels, int out_channels)
    {
        // Encoder
        enc1 = register_module("enc1", std::make_shared<DoubleConv>(in_channels, 64));
        enc2 = register_module("enc2", std::make_shared<DoubleConv>(64, 128));
        enc3 = register_module("enc3", std::make_shared<DoubleConv>(128, 256));

        // Bottleneck
        bottleneck = register_module("bottleneck", std::make_shared<DoubleConv>(256, 512));

        // Decoder
        upconv3 = register_module("upconv3", torch::nn::ConvTranspose2d(
                                      torch::nn::ConvTranspose2dOptions(512, 256, 2).stride(2)));
        dec3 = register_module("dec3", std::make_shared<DoubleConv>(512, 256));
        upconv2 = register_module("upconv2", torch::nn::ConvTranspose2d(
                                      torch::nn::ConvTranspose2dOptions(256, 128, 2).stride(2)));
        dec2 = register_module("dec2", std::make_shared<DoubleConv>(256, 128));
        upconv1 = register_module("upconv1", torch::nn::ConvTranspose2d(
                                      torch::nn::ConvTranspose2dOptions(128, 64, 2).stride(2)));
        dec1 = register_module("dec1", std::make_shared<DoubleConv>(128, 64));

        // Output layer
        out_conv = register_module("out_conv", torch::nn::Conv2d(
                                       torch::nn::Conv2dOptions(64, out_channels, 1)));
    }

    auto UNet::forward(std::initializer_list<std::any> tensors) -> std::any
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

    torch::Tensor UNet::forward(torch::Tensor x)
    {
        auto e1 = enc1->forward(x); // [batch, 64, 28, 28]
        auto p1 = torch::max_pool2d(e1, 2); // [batch, 64, 14, 14]
        auto e2 = enc2->forward(p1); // [batch, 128, 14, 14]
        auto p2 = torch::max_pool2d(e2, 2); // [batch, 128, 7, 7]
        auto e3 = enc3->forward(p2); // [batch, 256, 7, 7]
        auto p3 = torch::max_pool2d(e3, 2); // [batch, 256, 3, 3]

        // Bottleneck
        auto b = bottleneck->forward(p3); // [batch, 512, 3, 3]

        // Decoder
        auto u3 = upconv3->forward(b); // [batch, 256, 6, 6]
        // Pad to match e3 size (7x7)
        u3 = torch::pad(u3, {0, 1, 0, 1}); // [batch, 256, 7, 7]
        auto d3 = dec3->forward(torch::cat({u3, e3}, 1)); // [batch, 256, 7, 7]
        auto u2 = upconv2->forward(d3); // [batch, 128, 14, 14]
        auto d2 = dec2->forward(torch::cat({u2, e2}, 1)); // [batch, 128, 14, 14]
        auto u1 = upconv1->forward(d2); // [batch, 64, 28, 28]
        auto d1 = dec1->forward(torch::cat({u1, e1}, 1)); // [batch, 64, 28, 28]

        // Output
        auto out = out_conv->forward(d1); // [batch, out_channels, 28, 28]
        return out;
    }

    DiceLoss::DiceLoss(float smooth) : smooth_(smooth)
    {
    }

    auto DiceLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor input = tensor_vec[0];
        torch::Tensor target = tensor_vec[1];
        return this->forward(input, target);
    }

    torch::Tensor DiceLoss::forward(torch::Tensor input, torch::Tensor target)
    {
        // input: [batch, 1, h, w], target: [batch, 1, h, w]
        input = torch::sigmoid(input); // Convert logits to probabilities
        auto intersection = (input * target).sum({2, 3}); // [batch, 1]
        auto union1 = input.sum({2, 3}) + target.sum({2, 3}); // [batch, 1]
        auto dice = (2.0 * intersection + smooth_) / (union1 + smooth_); // [batch, 1]
        return 1.0 - dice.mean(); // Average loss
    }
}
