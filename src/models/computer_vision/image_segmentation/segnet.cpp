#include "include/models/computer_vision/image_segmentation/segnet.h"


using namespace std;

//SegNet GROK

// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // Encoder Block (Conv + BN + ReLU + MaxPool with Indices)
// struct EncoderBlockImpl : torch::nn::Module {
//     EncoderBlockImpl(int in_channels, int out_channels, bool pool = true) {
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(1).padding(1)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));
//         relu = register_module("relu", torch::nn::ReLU());
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(1)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
//         if (pool) {
//             maxpool = register_module("maxpool", torch::nn::MaxPool2d(
//                 torch::nn::MaxPool2dOptions(2).stride(2).return_indices(true)));
//         }
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
//         x = relu->forward(bn1->forward(conv1->forward(x)));
//         x = relu->forward(bn2->forward(conv2->forward(x)));
//         torch::Tensor indices = torch::empty({0}, x.options());
//         if (maxpool.defined()) {
//             auto [out, idx] = maxpool->forward(x);
//             x = out;
//             indices = idx;
//         }
//         return {x, indices};
//     }
//
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
//     torch::nn::ReLU relu{nullptr};
//     torch::nn::MaxPool2d maxpool{nullptr};
// };
// TORCH_MODULE(EncoderBlock);
//
// // Decoder Block (MaxUnpool + Conv + BN + ReLU)
// struct DecoderBlockImpl : torch::nn::Module {
//     DecoderBlockImpl(int in_channels, int out_channels, bool unpool = true) {
//         if (unpool) {
//             maxunpool = register_module("maxunpool", torch::nn::MaxUnpool2d(
//                 torch::nn::MaxUnpool2dOptions(2).stride(2)));
//         }
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, in_channels, 3).stride(1).padding(1)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(in_channels));
//         relu = register_module("relu", torch::nn::ReLU());
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(1).padding(1)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
//     }
//
//     torch::Tensor forward(torch::Tensor x, const torch::Tensor& indices, const std::vector<int64_t>& output_size) {
//         if (maxunpool.defined()) {
//             x = maxunpool->forward(x, indices, output_size);
//         }
//         x = relu->forward(bn1->forward(conv1->forward(x)));
//         x = relu->forward(bn2->forward(conv2->forward(x)));
//         return x;
//     }
//
//     torch::nn::MaxUnpool2d maxunpool{nullptr};
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(DecoderBlock);
//
// // SegNet Model
// struct SegNetImpl : torch::nn::Module {
//     SegNetImpl(int num_classes) : num_classes_(num_classes) {
//         // Encoder
//         encoder1 = register_module("encoder1", EncoderBlock(1, 64, true));    // [batch, 64, 14, 14]
//         encoder2 = register_module("encoder2", EncoderBlock(64, 128, true));  // [batch, 128, 7, 7]
//         encoder3 = register_module("encoder3", EncoderBlock(128, 256, true)); // [batch, 256, 4, 4]
//         encoder4 = register_module("encoder4", EncoderBlock(256, 512, false)); // [batch, 512, 4, 4]
//
//         // Decoder
//         decoder4 = register_module("decoder4", DecoderBlock(512, 256, false)); // [batch, 256, 4, 4]
//         decoder3 = register_module("decoder3", DecoderBlock(256, 128, true)); // [batch, 128, 7, 7]
//         decoder2 = register_module("decoder2", DecoderBlock(128, 64, true));  // [batch, 64, 14, 14]
//         decoder1 = register_module("decoder1", DecoderBlock(64, 64, true));   // [batch, 64, 28, 28]
//
//         // Classifier
//         classifier = register_module("classifier", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, num_classes, 1).stride(1)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Encoder
//         auto [e1, idx1] = encoder1->forward(x);   // [batch, 64, 14, 14]
//         auto [e2, idx2] = encoder2->forward(e1);  // [batch, 128, 7, 7]
//         auto [e3, idx3] = encoder3->forward(e2);  // [batch, 256, 4, 4]
//         auto [e4, idx4] = encoder4->forward(e3);  // [batch, 512, 4, 4]
//
//         // Decoder
//         auto d4 = decoder4->forward(e4, idx4, {4, 4});       // [batch, 256, 4, 4]
//         auto d3 = decoder3->forward(d4, idx3, {7, 7});       // [batch, 128, 7, 7]
//         auto d2 = decoder2->forward(d3, idx2, {14, 14});     // [batch, 64, 14, 14]
//         auto d1 = decoder1->forward(d2, idx1, {28, 28});     // [batch, 64, 28, 28]
//
//         // Classifier
//         auto output = classifier->forward(d1); // [batch, num_classes, 28, 28]
//
//         return output;
//     }
//
//     int num_classes_;
//     EncoderBlock encoder1{nullptr}, encoder2{nullptr}, encoder3{nullptr}, encoder4{nullptr};
//     DecoderBlock decoder1{nullptr}, decoder2{nullptr}, decoder3{nullptr}, decoder4{nullptr};
//     torch::nn::Conv2d classifier{nullptr};
// };
// TORCH_MODULE(SegNet);
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
//         SegNet model(num_classes);
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
//                 cv::imwrite("predicted_segnet_epoch_" + std::to_string(epoch + 1) + ".jpg", mask);
//             }
//         }
//
//         // Save model
//         torch::save(model, "segnet.pt");
//         std::cout << "Model saved as segnet.pt" << std::endl;
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
    SegNet::SegNet(int num_classes, int in_channels)
    {
    }

    SegNet::SegNet(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void SegNet::reset()
    {
    }

    auto SegNet::forward(std::initializer_list<std::any> tensors) -> std::any
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
