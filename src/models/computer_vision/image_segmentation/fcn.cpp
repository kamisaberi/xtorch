#include "include/models/computer_vision/image_segmentation/fcn.h"


using namespace std;


//FCN GROK

// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // VGG-like Backbone for FCN
// struct VGGBackboneImpl : torch::nn::Module {
//     VGGBackboneImpl() {
//         // Block 1: stride 1
//         conv1_1 = register_module("conv1_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(1, 64, 3).stride(1).padding(1))); // [batch, 64, 28, 28]
//         conv1_2 = register_module("conv1_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1))); // [batch, 64, 28, 28]
//
//         // Block 2: stride 2
//         conv2_1 = register_module("conv2_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1))); // [batch, 128, 28, 28]
//         conv2_2 = register_module("conv2_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1))); // [batch, 128, 28, 28]
//
//         // Block 3: stride 4
//         conv3_1 = register_module("conv3_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1))); // [batch, 256, 14, 14]
//         conv3_2 = register_module("conv3_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1))); // [batch, 256, 14, 14]
//         conv3_3 = register_module("conv3_3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1))); // [batch, 256, 14, 14]
//
//         // Block 4: stride 8
//         conv4_1 = register_module("conv4_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1))); // [batch, 512, 7, 7]
//         conv4_2 = register_module("conv4_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1))); // [batch, 512, 7, 7]
//         conv4_3 = register_module("conv4_3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1))); // [batch, 512, 7, 7]
//
//         relu = register_module("relu", torch::nn::ReLU());
//         pool = register_module("pool", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(2).stride(2)));
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
//         // Block 1
//         x = relu->forward(conv1_1->forward(x)); // [batch, 64, 28, 28]
//         x = relu->forward(conv1_2->forward(x)); // [batch, 64, 28, 28]
//         auto pool1 = x; // Save for skip connection (stride 1)
//         x = pool->forward(x); // [batch, 64, 14, 14]
//
//         // Block 2
//         x = relu->forward(conv2_1->forward(x)); // [batch, 128, 14, 14]
//         x = relu->forward(conv2_2->forward(x)); // [batch, 128, 14, 14]
//         auto pool2 = x; // Save for skip connection (stride 2)
//         x = pool->forward(x); // [batch, 128, 7, 7]
//
//         // Block 3
//         x = relu->forward(conv3_1->forward(x)); // [batch, 256, 7, 7]
//         x = relu->forward(conv3_2->forward(x)); // [batch, 256, 7, 7]
//         x = relu->forward(conv3_3->forward(x)); // [batch, 256, 7, 7]
//         auto pool3 = x; // Save for skip connection (stride 4)
//         x = pool->forward(x); // [batch, 256, 4, 4]
//
//         // Block 4
//         x = relu->forward(conv4_1->forward(x)); // [batch, 512, 4, 4]
//         x = relu->forward(conv4_2->forward(x)); // [batch, 512, 4, 4]
//         x = relu->forward(conv4_3->forward(x)); // [batch, 512, 4, 4]
//
//         return {x, pool3, pool2}; // Return features at stride 8, 4, 2
//     }
//
//     torch::nn::Conv2d conv1_1{nullptr}, conv1_2{nullptr}, conv2_1{nullptr}, conv2_2{nullptr};
//     torch::nn::Conv2d conv3_1{nullptr}, conv3_2{nullptr}, conv3_3{nullptr};
//     torch::nn::Conv2d conv4_1{nullptr}, conv4_2{nullptr}, conv4_3{nullptr};
//     torch::nn::ReLU relu{nullptr};
//     torch::nn::MaxPool2d pool{nullptr};
// };
// TORCH_MODULE(VGGBackbone);
//
// // FCN-8s Model
// struct FCN8sImpl : torch::nn::Module {
//     FCN8sImpl(int num_classes) : num_classes_(num_classes) {
//         backbone = register_module("backbone", VGGBackbone());
//
//         // Score layers for skip connections
//         score_pool4 = register_module("score_pool4", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, num_classes, 1).stride(1))); // Stride 4
//         score_pool3 = register_module("score_pool3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(128, num_classes, 1).stride(1))); // Stride 2
//
//         // Classifier for deepest features
//         classifier = register_module("classifier", torch::nn::Sequential(
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 4096, 1).stride(1)), // FC layer as 1x1 conv
//             torch::nn::ReLU(),
//             torch::nn::Dropout2d(0.5),
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(4096, 4096, 1).stride(1)),
//             torch::nn::ReLU(),
//             torch::nn::Dropout2d(0.5),
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(4096, num_classes, 1).stride(1))
//         ));
//
//         // Upsampling layers
//         upscore2 = register_module("upscore2", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(num_classes, num_classes, 4).stride(2).padding(1))); // 2x
//         upscore_pool4 = register_module("upscore_pool4", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(num_classes, num_classes, 4).stride(2).padding(1))); // 2x
//         upscore8 = register_module("upscore8", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(num_classes, num_classes, 8).stride(8).padding(0))); // 8x
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto input_size = std::vector<int64_t>{x.size(2), x.size(3)}; // [height, width]
//         auto [pool5, pool4, pool3] = backbone->forward(x); // [batch, 512, 4, 4], [batch, 256, 7, 7], [batch, 128, 14, 14]
//
//         // Score deepest features
//         auto score = classifier->forward(pool5); // [batch, num_classes, 4, 4]
//         score = upscore2->forward(score); // [batch, num_classes, 8, 8]
//
//         // Combine with pool4 (stride 4)
//         auto score_pool4_out = score_pool4->forward(pool4); // [batch, num_classes, 7, 7]
//         score = score + crop(score_pool4_out, score); // Align and add
//         score = upscore_pool4->forward(score); // [batch, num_classes, 14, 14]
//
//         // Combine with pool3 (stride 2)
//         auto score_pool3_out = score_pool3->forward(pool3); // [batch, num_classes, 14, 14]
//         score = score + crop(score_pool3_out, score); // Align and add
//         score = upscore8->forward(score); // [batch, num_classes, 112, 112]
//
//         // Crop to input size
//         score = crop(score, {input_size[0], input_size[1]});
//
//         return score; // [batch, num_classes, 28, 28]
//     }
//
//     torch::Tensor crop(torch::Tensor x, const torch::Tensor& target) {
//         int64_t h = target.size(2), w = target.size(3);
//         int64_t x_h = x.size(2), x_w = x.size(3);
//         int64_t start_h = (x_h - h) / 2, start_w = (x_w - w) / 2;
//         return x.slice(2, start_h, start_h + h).slice(3, start_w, start_w + w);
//     }
//
//     torch::Tensor crop(torch::Tensor x, const std::vector<int64_t>& size) {
//         int64_t h = size[0], w = size[1];
//         int64_t x_h = x.size(2), x_w = x.size(3);
//         int64_t start_h = (x_h - h) / 2, start_w = (x_w - w) / 2;
//         return x.slice(2, start_h, start_h + h).slice(3, start_w, start_w + w);
//     }
//
//     int num_classes_;
//     VGGBackbone backbone{nullptr};
//     torch::nn::Conv2d score_pool4{nullptr}, score_pool3{nullptr};
//     torch::nn::Sequential classifier{nullptr};
//     torch::nn::ConvTranspose2d upscore2{nullptr}, upscore_pool4{nullptr}, upscore8{nullptr};
// };
// TORCH_MODULE(FCN8s);
//
// // Dataset for Images and Segmentation Masks
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
//         torch::Tensor mask_tensor = torch::from_blob(mask.data, {mask.rows, mask.cols}, torch::kInt64);
//         mask_tensor = mask_tensor / 255; // Assume binary mask (0 or 255) to 0 or 1
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
//         const int num_classes = 2; // Binary segmentation (foreground/background)
//         const int batch_size = 16;
//         const float lr = 0.001;
//         const int num_epochs = 30;
//
//         // Initialize model
//         FCN8s model(num_classes);
//         model->to(device);
//
//         // Optimizer
//         torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));
//
//         // Loss function
//         auto ce_loss = torch::nn::CrossEntropyLoss();
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
//                 auto masks = batch.target.to(device); // [batch, 28, 28]
//
//                 optimizer.zero_grad();
//                 auto logits = model->forward(images); // [batch, num_classes, 28, 28]
//                 auto loss = ce_loss->forward(logits, masks);
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
//             // Save predicted mask every 10 epochs
//             if ((epoch + 1) % 10 == 0) {
//                 torch::NoGradGuard no_grad;
//                 auto sample_batch = *data_loader->begin();
//                 auto input_img = sample_batch.data[0].unsqueeze(0).to(device);
//                 auto pred_logits = model->forward(input_img); // [1, num_classes, 28, 28]
//                 auto pred_mask = torch::argmax(pred_logits, 1).squeeze().to(torch::kCPU); // [28, 28]
//                 cv::Mat mask(28, 28, CV_32S, pred_mask.data_ptr<int64_t>());
//                 mask.convertTo(mask, CV_8U, 255);
//                 cv::imwrite("predicted_fcn_epoch_" + std::to_string(epoch + 1) + ".jpg", mask);
//             }
//         }
//
//         // Save model
//         torch::save(model, "fcn.pt");
//         std::cout << "Model saved as fcn.pt" << std::endl;
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
    FCN::FCN(int num_classes, int in_channels)
    {
    }

    FCN::FCN(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void FCN::reset()
    {
    }

    auto FCN::forward(std::initializer_list<std::any> tensors) -> std::any
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
