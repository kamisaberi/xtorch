#include <models/computer_vision/image_segmentation/deep_lab.h>


using namespace std;
//DeepLAbV1 GROK


// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // VGG-like Backbone for DeepLabV1
// struct VGGBackboneImpl : torch::nn::Module {
//     VGGBackboneImpl() {
//         // Block 1
//         conv1_1 = register_module("conv1_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(1, 64, 3).stride(1).padding(1)));
//         conv1_2 = register_module("conv1_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)));
//
//         // Block 2
//         conv2_1 = register_module("conv2_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)));
//         conv2_2 = register_module("conv2_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)));
//
//         // Block 3
//         conv3_1 = register_module("conv3_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)));
//         conv3_2 = register_module("conv3_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)));
//         conv3_3 = register_module("conv3_3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)));
//
//         // Block 4 (with atrous convolution)
//         conv4_1 = register_module("conv4_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(2).dilation(2)));
//         conv4_2 = register_module("conv4_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(2).dilation(2)));
//         conv4_3 = register_module("conv4_3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(2).dilation(2)));
//
//         // Block 5 (with atrous convolution)
//         conv5_1 = register_module("conv5_1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(4).dilation(4)));
//         conv5_2 = register_module("conv5_2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(4).dilation(4)));
//         conv5_3 = register_module("conv5_3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(4).dilation(4)));
//
//         relu = register_module("relu", torch::nn::ReLU());
//         pool = register_module("pool", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(2).stride(2)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Block 1
//         x = relu->forward(conv1_1->forward(x)); // [batch, 64, 28, 28]
//         x = relu->forward(conv1_2->forward(x)); // [batch, 64, 28, 28]
//         x = pool->forward(x); // [batch, 64, 14, 14]
//
//         // Block 2
//         x = relu->forward(conv2_1->forward(x)); // [batch, 128, 14, 14]
//         x = relu->forward(conv2_2->forward(x)); // [batch, 128, 14, 14]
//         x = pool->forward(x); // [batch, 128, 7, 7]
//
//         // Block 3
//         x = relu->forward(conv3_1->forward(x)); // [batch, 256, 7, 7]
//         x = relu->forward(conv3_2->forward(x)); // [batch, 256, 7, 7]
//         x = relu->forward(conv3_3->forward(x)); // [batch, 256, 7, 7]
//
//         // Block 4 (atrous)
//         x = relu->forward(conv4_1->forward(x)); // [batch, 512, 7, 7]
//         x = relu->forward(conv4_2->forward(x)); // [batch, 512, 7, 7]
//         x = relu->forward(conv4_3->forward(x)); // [batch, 512, 7, 7]
//
//         // Block 5 (atrous)
//         x = relu->forward(conv5_1->forward(x)); // [batch, 512, 7, 7]
//         x = relu->forward(conv5_2->forward(x)); // [batch, 512, 7, 7]
//         x = relu->forward(conv5_3->forward(x)); // [batch, 512, 7, 7]
//
//         return x;
//     }
//
//     torch::nn::Conv2d conv1_1{nullptr}, conv1_2{nullptr}, conv2_1{nullptr}, conv2_2{nullptr};
//     torch::nn::Conv2d conv3_1{nullptr}, conv3_2{nullptr}, conv3_3{nullptr};
//     torch::nn::Conv2d conv4_1{nullptr}, conv4_2{nullptr}, conv4_3{nullptr};
//     torch::nn::Conv2d conv5_1{nullptr}, conv5_2{nullptr}, conv5_3{nullptr};
//     torch::nn::ReLU relu{nullptr};
//     torch::nn::MaxPool2d pool{nullptr};
// };
// TORCH_MODULE(VGGBackbone);
//
// // DeepLabV1 Model
// struct DeepLabV1Impl : torch::nn::Module {
//     DeepLabV1Impl(int num_classes) : num_classes_(num_classes) {
//         backbone = register_module("backbone", VGGBackbone());
//         classifier = register_module("classifier", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(512, num_classes, 1).stride(1)));
//         upsample = register_module("upsample", torch::nn::Functional(
//             [](torch::Tensor x, const std::vector<int64_t>& size) {
//                 return torch::upsample_bilinear2d(x, size, /*align_corners=*/true);
//             }));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto input_size = std::vector<int64_t>{x.size(2), x.size(3)}; // [height, width]
//         x = backbone->forward(x); // [batch, 512, 7, 7]
//         x = classifier->forward(x); // [batch, num_classes, 7, 7]
//         x = upsample->forward(x, {input_size[0], input_size[1]}); // [batch, num_classes, 28, 28]
//
//         // Simple CRF approximation: Apply a Gaussian filter to refine boundaries
//         x = gaussian_smooth(x);
//
//         return x;
//     }
//
//     torch::Tensor gaussian_smooth(torch::Tensor x) {
//         // Approximate CRF with a Gaussian filter (simplified)
//         auto kernel = gaussian_kernel(3, 1.0).to(x.device());
//         kernel = kernel.repeat({x.size(1), 1, 1, 1}); // [num_classes, 1, 3, 3]
//         return torch::conv2d(x, kernel, /*bias=*/{}, /*stride=*/1, /*padding=*/1);
//     }
//
//     static torch::Tensor gaussian_kernel(int size, float sigma) {
//         auto grid = torch::arange(-size / 2, size / 2 + 1, torch::kFloat32);
//         auto x = grid.repeat({size, 1});
//         auto y = x.t();
//         auto kernel = torch::exp(-(x.pow(2) + y.pow(2)) / (2 * sigma * sigma));
//         kernel = kernel / kernel.sum();
//         return kernel.view({1, 1, size, size});
//     }
//
//     int num_classes_;
//     VGGBackbone backbone{nullptr};
//     torch::nn::Conv2d classifier{nullptr};
//     torch::nn::Functional upsample{nullptr};
// };
// TORCH_MODULE(DeepLabV1);
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
//         DeepLabV1 model(num_classes);
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
//                 cv::imwrite("predicted_deeplabv1_epoch_" + std::to_string(epoch + 1) + ".jpg", mask);
//             }
//         }
//
//         // Save model
//         torch::save(model, "deeplabv1.pt");
//         std::cout << "Model saved as deeplabv1.pt" << std::endl;
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
//
//     return 0;
// }


//DeepLabV2 GROK

// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // Residual Block for ResNet
// struct ResidualBlockImpl : torch::nn::Module {
//     ResidualBlockImpl(int in_channels, int out_channels, int stride = 1, int dilation = 1) {
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(dilation).dilation(dilation)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(dilation).dilation(dilation)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
//         relu = register_module("relu", torch::nn::ReLU());
//
//         if (stride != 1 || in_channels != out_channels) {
//             shortcut = register_module("shortcut", torch::nn::Conv2d(
//                 torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride)));
//             bn_shortcut = register_module("bn_shortcut", torch::nn::BatchNorm2d(out_channels));
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto residual = x;
//         x = relu->forward(bn1->forward(conv1->forward(x)));
//         x = bn2->forward(conv2->forward(x));
//
//         if (shortcut.defined()) {
//             residual = bn_shortcut->forward(shortcut->forward(residual));
//         }
//
//         x = relu->forward(x + residual);
//         return x;
//     }
//
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, shortcut{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn_shortcut{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(ResidualBlock);
//
// // Simplified ResNet Backbone
// struct ResNetBackboneImpl : torch::nn::Module {
//     ResNetBackboneImpl() {
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(1, 64, 7).stride(2).padding(3))); // [batch, 64, 14, 14]
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
//         relu = register_module("relu", torch::nn::ReLU());
//         pool = register_module("pool", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(3).stride(2).padding(1))); // [batch, 64, 7, 7]
//
//         // Layer 1
//         layer1->push_back("block1_1", ResidualBlock(64, 64));
//         layer1->push_back("block1_2", ResidualBlock(64, 64));
//
//         // Layer 2 (atrous)
//         layer2->push_back("block2_1", ResidualBlock(64, 128, 1, 2));
//         layer2->push_back("block2_2", ResidualBlock(128, 128, 1, 2));
//
//         // Layer 3 (atrous)
//         layer3->push_back("block3_1", ResidualBlock(128, 256, 1, 4));
//         layer3->push_back("block3_2", ResidualBlock(256, 256, 1, 4));
//
//         layer1 = register_module("layer1", layer1);
//         layer2 = register_module("layer2", layer2);
//         layer3 = register_module("layer3", layer3);
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = relu->forward(bn1->forward(conv1->forward(x))); // [batch, 64, 14, 14]
//         x = pool->forward(x); // [batch, 64, 7, 7]
//         x = layer1->forward(x); // [batch, 64, 7, 7]
//         x = layer2->forward(x); // [batch, 128, 7, 7]
//         x = layer3->forward(x); // [batch, 256, 7, 7]
//         return x;
//     }
//
//     torch::nn::Conv2d conv1{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr};
//     torch::nn::ReLU relu{nullptr};
//     torch::nn::MaxPool2d pool{nullptr};
//     torch::nn::Sequential layer1{nullptr}, layer2{nullptr}, layer3{nullptr};
// };
// TORCH_MODULE(ResNetBackbone);
//
// // Atrous Spatial Pyramid Pooling (ASPP) Module
// struct ASPPImpl : torch::nn::Module {
//     ASPPImpl(int in_channels, int out_channels, const std::vector<int>& dilations) {
//         for (size_t i = 0; i < dilations.size(); ++i) {
//             auto conv = torch::nn::Conv2d(
//                 torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(1).padding(dilations[i]).dilation(dilations[i]));
//             auto bn = torch::nn::BatchNorm2d(out_channels);
//             branches->push_back("conv_" + std::to_string(i), conv);
//             branches->push_back("bn_" + std::to_string(i), bn);
//         }
//         branches = register_module("branches", branches);
//         relu = register_module("relu", torch::nn::ReLU());
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         std::vector<torch::Tensor> features;
//         for (size_t i = 0; i < branches->size() / 2; ++i) {
//             auto feat = branches[i * 2]->forward(x);
//             feat = relu->forward(branches[i * 2 + 1]->forward(feat));
//             features.push_back(feat);
//         }
//         return torch::cat(features, 1); // Concatenate along channel dimension
//     }
//
//     torch::nn::Sequential branches{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(ASPP);
//
// // DeepLabV2 Model
// struct DeepLabV2Impl : torch::nn::Module {
//     DeepLabV2Impl(int num_classes) : num_classes_(num_classes) {
//         backbone = register_module("backbone", ResNetBackbone());
//         aspp = register_module("aspp", ASPP(256, 64, {1, 2, 4})); // Dilations for ASPP
//         classifier = register_module("classifier", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64 * 3, num_classes, 1).stride(1))); // 3 branches in ASPP
//         upsample = register_module("upsample", torch::nn::Functional(
//             [](torch::Tensor x, const std::vector<int64_t>& size) {
//                 return torch::upsample_bilinear2d(x, size, /*align_corners=*/true);
//             }));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto input_size = std::vector<int64_t>{x.size(2), x.size(3)}; // [height, width]
//         x = backbone->forward(x); // [batch, 256, 7, 7]
//         x = aspp->forward(x); // [batch, 64 * 3, 7, 7]
//         x = classifier->forward(x); // [batch, num_classes, 7, 7]
//         x = upsample->forward(x, {input_size[0], input_size[1]}); // [batch, num_classes, 28, 28]
//
//         // Simple CRF approximation: Apply a Gaussian filter
//         x = gaussian_smooth(x);
//
//         return x;
//     }
//
//     torch::Tensor gaussian_smooth(torch::Tensor x) {
//         // Approximate CRF with a Gaussian filter
//         auto kernel = gaussian_kernel(3, 1.0).to(x.device());
//         kernel = kernel.repeat({x.size(1), 1, 1, 1}); // [num_classes, 1, 3, 3]
//         return torch::conv2d(x, kernel, /*bias=*/{}, /*stride=*/1, /*padding=*/1);
//     }
//
//     static torch::Tensor gaussian_kernel(int size, float sigma) {
//         auto grid = torch::arange(-size / 2, size / 2 + 1, torch::kFloat32);
//         auto x = grid.repeat({size, 1});
//         auto y = x.t();
//         auto kernel = torch::exp(-(x.pow(2) + y.pow(2)) / (2 * sigma * sigma));
//         kernel = kernel / kernel.sum();
//         return kernel.view({1, 1, size, size});
//     }
//
//     int num_classes_;
//     ResNetBackbone backbone{nullptr};
//     ASPP aspp{nullptr};
//     torch::nn::Conv2d classifier{nullptr};
//     torch::nn::Functional upsample{nullptr};
// };
// TORCH_MODULE(DeepLabV2);
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
//         DeepLabV2 model(num_classes);
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
//                 cv::imwrite("predicted_deeplabv2_epoch_" + std::to_string(epoch + 1) + ".jpg", mask);
//             }
//         }
//
//         // Save model
//         torch::save(model, "deeplabv2.pt");
//         std::cout << "Model saved as deeplabv2.pt" << std::endl;
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
//
//     return 0;
// }

//DeepLabV3 GROK

// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // Residual Block for ResNet
// struct ResidualBlockImpl : torch::nn::Module {
//     ResidualBlockImpl(int in_channels, int out_channels, int stride = 1, int dilation = 1) {
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(dilation).dilation(dilation)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(dilation).dilation(dilation)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
//         relu = register_module("relu", torch::nn::ReLU());
//
//         if (stride != 1 || in_channels != out_channels) {
//             shortcut = register_module("shortcut", torch::nn::Conv2d(
//                 torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride)));
//             bn_shortcut = register_module("bn_shortcut", torch::nn::BatchNorm2d(out_channels));
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto residual = x;
//         x = relu->forward(bn1->forward(conv1->forward(x)));
//         x = bn2->forward(conv2->forward(x));
//
//         if (shortcut.defined()) {
//             residual = bn_shortcut->forward(shortcut->forward(residual));
//         }
//
//         x = relu->forward(x + residual);
//         return x;
//     }
//
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, shortcut{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn_shortcut{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(ResidualBlock);
//
// // Simplified ResNet Backbone
// struct ResNetBackboneImpl : torch::nn::Module {
//     ResNetBackboneImpl() {
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(1, 64, 7).stride(2).padding(3))); // [batch, 64, 14, 14]
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
//         relu = register_module("relu", torch::nn::ReLU());
//         pool = register_module("pool", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(3).stride(2).padding(1))); // [batch, 64, 7, 7]
//
//         // Layer 1
//         layer1->push_back("block1_1", ResidualBlock(64, 64));
//         layer1->push_back("block1_2", ResidualBlock(64, 64));
//
//         // Layer 2 (atrous)
//         layer2->push_back("block2_1", ResidualBlock(64, 128, 1, 2));
//         layer2->push_back("block2_2", ResidualBlock(128, 128, 1, 2));
//
//         // Layer 3 (atrous)
//         layer3->push_back("block3_1", ResidualBlock(128, 256, 1, 4));
//         layer3->push_back("block3_2", ResidualBlock(256, 256, 1, 4));
//
//         // Layer 4 (atrous)
//         layer4->push_back("block4_1", ResidualBlock(256, 512, 1, 8));
//         layer4->push_back("block4_2", ResidualBlock(512, 512, 1, 8));
//
//         layer1 = register_module("layer1", layer1);
//         layer2 = register_module("layer2", layer2);
//         layer3 = register_module("layer3", layer3);
//         layer4 = register_module("layer4", layer4);
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = relu->forward(bn1->forward(conv1->forward(x))); // [batch, 64, 14, 14]
//         x = pool->forward(x); // [batch, 64, 7, 7]
//         x = layer1->forward(x); // [batch, 64, 7, 7]
//         x = layer2->forward(x); // [batch, 128, 7, 7]
//         x = layer3->forward(x); // [batch, 256, 7, 7]
//         x = layer4->forward(x); // [batch, 512, 7, 7]
//         return x;
//     }
//
//     torch::nn::Conv2d conv1{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr};
//     torch::nn::ReLU relu{nullptr};
//     torch::nn::MaxPool2d pool{nullptr};
//     torch::nn::Sequential layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
// };
// TORCH_MODULE(ResNetBackbone);
//
// // Enhanced Atrous Spatial Pyramid Pooling (ASPP) Module
// struct ASPPImpl : torch::nn::Module {
//     ASPPImpl(int in_channels, int out_channels, const std::vector<int>& dilations) {
//         // Atrous convolution branches
//         for (size_t i = 0; i < dilations.size(); ++i) {
//             auto conv = torch::nn::Conv2d(
//                 torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(1).padding(dilations[i]).dilation(dilations[i]));
//             auto bn = torch::nn::BatchNorm2d(out_channels);
//             branches->push_back("conv_" + std::to_string(i), conv);
//             branches->push_back("bn_" + std::to_string(i), bn);
//         }
//
//         // Global average pooling branch
//         global_pool = register_module("global_pool", torch::nn::Sequential(
//             torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})),
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(1)),
//             torch::nn::BatchNorm2d(out_channels)
//         ));
//
//         branches = register_module("branches", branches);
//         relu = register_module("relu", torch::nn::ReLU());
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         std::vector<torch::Tensor> features;
//         // Atrous branches
//         for (size_t i = 0; i < branches->size() / 2; ++i) {
//             auto feat = branches[i * 2]->forward(x);
//             feat = relu->forward(branches[i * 2 + 1]->forward(feat));
//             features.push_back(feat);
//         }
//
//         // Global pooling branch
//         auto global_feat = global_pool->forward(x); // [batch, out_channels, 1, 1]
//         global_feat = torch::upsample_bilinear2d(global_feat, {x.size(2), x.size(3)}, /*align_corners=*/true);
//         features.push_back(relu->forward(global_feat));
//
//         return torch::cat(features, 1); // Concatenate along channel dimension
//     }
//
//     torch::nn::Sequential branches{nullptr}, global_pool{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(ASPP);
//
// // DeepLabV3 Model
// struct DeepLabV3Impl : torch::nn::Module {
//     DeepLabV3Impl(int num_classes) : num_classes_(num_classes) {
//         backbone = register_module("backbone", ResNetBackbone());
//         aspp = register_module("aspp", ASPP(512, 64, {1, 2, 4, 6})); // Dilations for ASPP
//         classifier = register_module("classifier", torch::nn::Sequential(
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(64 * 4, 256, 3).stride(1).padding(1)), // 4 branches (3 atrous + 1 global)
//             torch::nn::BatchNorm2d(256),
//             torch::nn::ReLU(),
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(256, num_classes, 1).stride(1))
//         ));
//         upsample = register_module("upsample", torch::nn::Functional(
//             [](torch::Tensor x, const std::vector<int64_t>& size) {
//                 return torch::upsample_bilinear2d(x, size, /*align_corners=*/true);
//             }));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto input_size = std::vector<int64_t>{x.size(2), x.size(3)}; // [height, width]
//         x = backbone->forward(x); // [batch, 512, 7, 7]
//         x = aspp->forward(x); // [batch, 64 * 4, 7, 7]
//         x = classifier->forward(x); // [batch, num_classes, 7, 7]
//         x = upsample->forward(x, {input_size[0], input_size[1]}); // [batch, num_classes, 28, 28]
//         return x;
//     }
//
//     int num_classes_;
//     ResNetBackbone backbone{nullptr};
//     ASPP aspp{nullptr};
//     torch::nn::Sequential classifier{nullptr};
//     torch::nn::Functional upsample{nullptr};
// };
// TORCH_MODULE(DeepLabV3);
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
//         DeepLabV3 model(num_classes);
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
//                 cv::imwrite("predicted_deeplabv3_epoch_" + std::to_string(epoch + 1) + ".jpg", mask);
//             }
//         }
//
//         // Save model
//         torch::save(model, "deeplabv3.pt");
//         std::cout << "Model saved as deeplabv3.pt" << std::endl;
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
//
//     return 0;
// }

//DeepLabV3Plus GROK

// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // Residual Block for ResNet
// struct ResidualBlockImpl : torch::nn::Module {
//     ResidualBlockImpl(int in_channels, int out_channels, int stride = 1, int dilation = 1) {
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(dilation).dilation(dilation)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(dilation).dilation(dilation)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
//         relu = register_module("relu", torch::nn::ReLU());
//
//         if (stride != 1 || in_channels != out_channels) {
//             shortcut = register_module("shortcut", torch::nn::Conv2d(
//                 torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride)));
//             bn_shortcut = register_module("bn_shortcut", torch::nn::BatchNorm2d(out_channels));
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto residual = x;
//         x = relu->forward(bn1->forward(conv1->forward(x)));
//         x = bn2->forward(conv2->forward(x));
//
//         if (shortcut.defined()) {
//             residual = bn_shortcut->forward(shortcut->forward(residual));
//         }
//
//         x = relu->forward(x + residual);
//         return x;
//     }
//
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, shortcut{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn_shortcut{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(ResidualBlock);
//
// // Simplified ResNet Backbone
// struct ResNetBackboneImpl : torch::nn::Module {
//     ResNetBackboneImpl() {
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(1, 64, 7).stride(2).padding(3))); // [batch, 64, 14, 14]
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
//         relu = register_module("relu", torch::nn::ReLU());
//         pool = register_module("pool", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(3).stride(2).padding(1))); // [batch, 64, 7, 7]
//
//         // Layer 1
//         layer1->push_back("block1_1", ResidualBlock(64, 64));
//         layer1->push_back("block1_2", ResidualBlock(64, 64));
//
//         // Layer 2 (atrous)
//         layer2->push_back("block2_1", ResidualBlock(64, 128, 1, 2));
//         layer2->push_back("block2_2", ResidualBlock(128, 128, 1, 2));
//
//         // Layer 3 (atrous)
//         layer3->push_back("block3_1", ResidualBlock(128, 256, 1, 4));
//         layer3->push_back("block3_2", ResidualBlock(256, 256, 1, 4));
//
//         // Layer 4 (atrous)
//         layer4->push_back("block4_1", ResidualBlock(256, 512, 1, 8));
//         layer4->push_back("block4_2", ResidualBlock(512, 512, 1, 8));
//
//         layer1 = register_module("layer1", layer1);
//         layer2 = register_module("layer2", layer2);
//         layer3 = register_module("layer3", layer3);
//         layer4 = register_module("layer4", layer4);
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
//         x = relu->forward(bn1->forward(conv1->forward(x))); // [batch, 64, 14, 14]
//         auto low_level_features = x; // Save for decoder
//         x = pool->forward(x); // [batch, 64, 7, 7]
//         x = layer1->forward(x); // [batch, 64, 7, 7]
//         x = layer2->forward(x); // [batch, 128, 7, 7]
//         x = layer3->forward(x); // [batch, 256, 7, 7]
//         x = layer4->forward(x); // [batch, 512, 7, 7]
//         return {x, low_level_features};
//     }
//
//     torch::nn::Conv2d conv1{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr};
//     torch::nn::ReLU relu{nullptr};
//     torch::nn::MaxPool2d pool{nullptr};
//     torch::nn::Sequential layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
// };
// TORCH_MODULE(ResNetBackbone);
//
// // Enhanced Atrous Spatial Pyramid Pooling (ASPP) Module
// struct ASPPImpl : torch::nn::Module {
//     ASPPImpl(int in_channels, int out_channels, const std::vector<int>& dilations) {
//         // Atrous convolution branches
//         for (size_t i = 0; i < dilations.size(); ++i) {
//             auto conv = torch::nn::Conv2d(
//                 torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(1).padding(dilations[i]).dilation(dilations[i]));
//             auto bn = torch::nn::BatchNorm2d(out_channels);
//             branches->push_back("conv_" + std::to_string(i), conv);
//             branches->push_back("bn_" + std::to_string(i), bn);
//         }
//
//         // Global average pooling branch
//         global_pool = register_module("global_pool", torch::nn::Sequential(
//             torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})),
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(1)),
//             torch::nn::BatchNorm2d(out_channels)
//         ));
//
//         branches = register_module("branches", branches);
//         relu = register_module("relu", torch::nn::ReLU());
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         std::vector<torch::Tensor> features;
//         // Atrous branches
//         for (size_t i = 0; i < branches->size() / 2; ++i) {
//             auto feat = branches[i * 2]->forward(x);
//             feat = relu->forward(branches[i * 2 + 1]->forward(feat));
//             features.push_back(feat);
//         }
//
//         // Global pooling branch
//         auto global_feat = global_pool->forward(x); // [batch, out_channels, 1, 1]
//         global_feat = torch::upsample_bilinear2d(global_feat, {x.size(2), x.size(3)}, /*align_corners=*/true);
//         features.push_back(relu->forward(global_feat));
//
//         return torch::cat(features, 1); // Concatenate along channel dimension
//     }
//
//     torch::nn::Sequential branches{nullptr}, global_pool{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(ASPP);
//
// // Decoder Module
// struct DecoderImpl : torch::nn::Module {
//     DecoderImpl(int in_channels, int low_level_channels, int out_channels) {
//         low_level_conv = register_module("low_level_conv", torch::nn::Sequential(
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(low_level_channels, 48, 1).stride(1)),
//             torch::nn::BatchNorm2d(48),
//             torch::nn::ReLU()
//         ));
//
//         decoder_conv = register_module("decoder_conv", torch::nn::Sequential(
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels + 48, 256, 3).stride(1).padding(1)),
//             torch::nn::BatchNorm2d(256),
//             torch::nn::ReLU(),
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
//             torch::nn::BatchNorm2d(256),
//             torch::nn::ReLU(),
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(256, out_channels, 1).stride(1))
//         ));
//
//         upsample = register_module("upsample", torch::nn::Functional(
//             [](torch::Tensor x, const std::vector<int64_t>& size) {
//                 return torch::upsample_bilinear2d(x, size, /*align_corners=*/true);
//             }));
//     }
//
//     torch::Tensor forward(torch::Tensor aspp_features, torch::Tensor low_level_features, const std::vector<int64_t>& output_size) {
//         // Process low-level features
//         auto low_level = low_level_conv->forward(low_level_features); // [batch, 48, 14, 14]
//
//         // Upsample ASPP features
//         aspp_features = upsample->forward(aspp_features, {low_level.size(2), low_level.size(3)}); // [batch, in_channels, 14, 14]
//
//         // Concatenate and process
//         auto x = torch::cat({aspp_features, low_level}, 1); // [batch, in_channels+48, 14, 14]
//         x = decoder_conv->forward(x); // [batch, out_channels, 14, 14]
//
//         // Final upsampling
//         x = upsample->forward(x, output_size); // [batch, out_channels, 28, 28]
//
//         return x;
//     }
//
//     torch::nn::Sequential low_level_conv{nullptr}, decoder_conv{nullptr};
//     torch::nn::Functional upsample{nullptr};
// };
// TORCH_MODULE(Decoder);
//
// // DeepLabV3Plus Model
// struct DeepLabV3PlusImpl : torch::nn::Module {
//     DeepLabV3PlusImpl(int num_classes) : num_classes_(num_classes) {
//         backbone = register_module("backbone", ResNetBackbone());
//         aspp = register_module("aspp", ASPP(512, 64, {6, 4, 2})); // Dilations for ASPP
//         decoder = register_module("decoder", Decoder(64 * 4, 64, num_classes)); // 4 ASPP branches
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto input_size = std::vector<int64_t>{x.size(2), x.size(3)}; // [height, width]
//         auto [encoder_features, low_level_features] = backbone->forward(x); // [batch, 512, 7, 7], [batch, 64, 14, 14]
//         auto aspp_features = aspp->forward(encoder_features); // [batch, 64 * 4, 7, 7]
//         auto output = decoder->forward(aspp_features, low_level_features, input_size); // [batch, num_classes, 28, 28]
//         return output;
//     }
//
//     int num_classes_;
//     ResNetBackbone backbone{nullptr};
//     ASPP aspp{nullptr};
//     Decoder decoder{nullptr};
// };
// TORCH_MODULE(DeepLabV3Plus);
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
//         DeepLabV3Plus model(num_classes);
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
//                 cv::imwrite("predicted_deeplabv3plus_epoch_" + std::to_string(epoch + 1) + ".jpg", mask);
//             }
//         }
//
//         // Save model
//         torch::save(model, "deeplabv3plus.pt");
//         std::cout << "Model saved as deeplabv3plus.pt" << std::endl;
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
    DeepLabV1::DeepLabV1(int num_classes, int in_channels)
    {
    }

    DeepLabV1::DeepLabV1(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DeepLabV1::reset()
    {
    }

    auto DeepLabV1::forward(std::initializer_list<std::any> tensors) -> std::any
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


    DeepLabV2::DeepLabV2(int num_classes, int in_channels)
    {
    }

    DeepLabV2::DeepLabV2(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DeepLabV2::reset()
    {
    }

    auto DeepLabV2::forward(std::initializer_list<std::any> tensors) -> std::any
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






    DeepLabV3::DeepLabV3(int num_classes, int in_channels)
    {
    }

    DeepLabV3::DeepLabV3(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DeepLabV3::reset()
    {
    }

    auto DeepLabV3::forward(std::initializer_list<std::any> tensors) -> std::any
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



    DeepLabV3Plus::DeepLabV3Plus(int num_classes, int in_channels)
    {
    }

    DeepLabV3Plus::DeepLabV3Plus(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DeepLabV3Plus::reset()
    {
    }

    auto DeepLabV3Plus::forward(std::initializer_list<std::any> tensors) -> std::any
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
