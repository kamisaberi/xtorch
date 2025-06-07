#include "include/models/computer_vision/image_segmentation/panet.h"


using namespace std;

//PANet GROK


#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>

// // Basic Residual Block for ResNet Backbone
// struct ResidualBlockImpl : torch::nn::Module {
//     ResidualBlockImpl(int in_channels, int out_channels, int stride = 1) {
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));
//         relu = register_module("relu", torch::nn::ReLU());
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(1)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
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
// // FPN Backbone with Top-Down Path
// struct FPNBackboneImpl : torch::nn::Module {
//     FPNBackboneImpl() {
//         // Initial convolution
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(1, 64, 7).stride(2).padding(3))); // [batch, 64, 14, 14]
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
//         relu = register_module("relu", torch::nn::ReLU());
//         pool = register_module("pool", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(2).stride(2))); // [batch, 64, 7, 7]
//
//         // ResNet layers
//         layer1->push_back("block1_1", ResidualBlock(64, 64)); // C2: [batch, 64, 7, 7]
//         layer1->push_back("block1_2", ResidualBlock(64, 64));
//         layer2->push_back("block2_1", ResidualBlock(64, 128, 2)); // C3: [batch, 128, 4, 4]
//         layer2->push_back("block2_2", ResidualBlock(128, 128));
//         layer3->push_back("block3_1", ResidualBlock(128, 256, 2)); // C4: [batch, 256, 2, 2]
//         layer3->push_back("block3_2", ResidualBlock(256, 256));
//         layer4->push_back("block4_1", ResidualBlock(256, 512, 2)); // C5: [batch, 512, 1, 1]
//         layer4->push_back("block4_2", ResidualBlock(512, 512));
//
//         // FPN top-down path
//         lateral5 = register_module("lateral5", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(512, 256, 1).stride(1))); // P5
//         lateral4 = register_module("lateral4", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, 1).stride(1))); // P4
//         lateral3 = register_module("lateral3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(128, 256, 1).stride(1))); // P3
//         lateral2 = register_module("lateral2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 256, 1).stride(1))); // P2
//
//         fpn5 = register_module("fpn5", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)));
//         fpn4 = register_module("fpn4", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)));
//         fpn3 = register_module("fpn3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)));
//         fpn2 = register_module("fpn2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)));
//
//         layer1 = register_module("layer1", layer1);
//         layer2 = register_module("layer2", layer2);
//         layer3 = register_module("layer3", layer3);
//         layer4 = register_module("layer4", layer4);
//     }
//
//     std::vector<torch::Tensor> forward(torch::Tensor x) {
//         // Backbone
//         x = relu->forward(bn1->forward(conv1->forward(x))); // [batch, 64, 14, 14]
//         x = pool->forward(x); // [batch, 64, 7, 7]
//         auto c2 = layer1->forward(x); // [batch, 64, 7, 7]
//         auto c3 = layer2->forward(c2); // [batch, 128, 4, 4]
//         auto c4 = layer3->forward(c3); // [batch, 256, 2, 2]
//         auto c5 = layer4->forward(c4); // [batch, 512, 1, 1]
//
//         // FPN top-down path
//         auto p5 = lateral5->forward(c5); // [batch, 256, 1, 1]
//         p5 = fpn5->forward(p5); // [batch, 256, 1, 1]
//
//         auto p4 = lateral4->forward(c4); // [batch, 256, 2, 2]
//         p4 = p4 + torch::upsample_nearest2d(p5, {2, 2}); // [batch, 256, 2, 2]
//         p4 = fpn4->forward(p4); // [batch, 256, 2, 2]
//
//         auto p3 = lateral3->forward(c3); // [batch, 256, 4, 4]
//         p3 = p3 + torch::upsample_nearest2d(p4, {4, 4}); // [batch, 256, 4, 4]
//         p3 = fpn3->forward(p3); // [batch, 256, 4, 4]
//
//         auto p2 = lateral2->forward(c2); // [batch, 256, 7, 7]
//         p2 = p2 + torch::upsample_nearest2d(p3, {8, 8}); // [batch, 256, 7, 7]
//         p2 = fpn2->forward(p2); // [batch, 256, 7, 7]
//
//         return {p2, p3, p4, p5, c2, c3, c4, c5}; // P2-P5, C2-C5 for bottom-up path
//     }
//
//     torch::nn::Conv2d conv1{nullptr}, lateral5{nullptr}, lateral4{nullptr}, lateral3{nullptr}, lateral2{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr};
//     torch::nn::ReLU relu{nullptr};
//     torch::nn::MaxPool2d pool{nullptr};
//     torch::nn::Sequential layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
//     torch::nn::Conv2d(nn::f, p5);
//     torch::nn::Conv2d(nn::f, nn::p4);
//     torch::nn::Conv2d(nn::f, nn::p3));
//     torch::nn::Conv2d(nn::f, nn::p2n));
// };
// TORCH_MODULE(FPNBackbone);
//
// // Bottom-Up Path Augmentation
// struct BottomUpPathImpl : torch::nn::Module {
//     BottomUpPathImpl() {
//         // Lateral connections
//         lateral_n2 = register_module("lateral_n2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, 1).stride(1)));
//         lateral_n3 = register_module("lateral_n3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, 1).stride(1)));
//         lateral_n4 = register_module("lateral_n4", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, 1).stride(1)));
//
//         // Bottom-up convolutions
//         conv_n2 = register_module("conv_n2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)));
//         conv_n3 = register_module("conv_n3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)));
//         conv_n4 = register_module("conv_n4", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)));
//
//         relu = register_module("relu", torch::nn::ReLU());
//     }
//
//     std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) {
//         auto p2 = inputs[0]; // [batch, 256, 7, 7]
//         auto p3 = inputs[1]; // [batch, 256, 4, 4]
//         auto p4 = inputs[2]; // [batch, 256, 2, 2]
//         auto p5 = inputs[3]; // [batch, 256, 1, 1]
//
//         // Bottom-up path
//         auto n2 = lateral_n2->forward(p2); // [batch, 256, 7, 7]
//         n2 = relu->forward(conv_n2->forward(n2)); // [batch, 256, 7, 7]
//
//         auto n3 = lateral_n3->forward(p3); // [batch, 256, 4, 4]
//         n3 += torch::max_pool2d(n2, 2, 2); // Downsample n2 to n3 size
//         n3 = relu->forward(conv_n3->forward(n3)); // [batch, 256, 4, 4]
//
//         auto n4 = lateral_n4->forward(p4); // [batch, 256, 2, 2]
//         n4 += torch::max_pool2d(n3, 2, 2); // Downsample n3 to n4 size
//         n4 = relu->forward(conv_n4->forward(n4)); // [batch, 256, 2, 2]
//
//         return {n2, n3, n4, p5}; // Return N2-N4, P5
//     }
//
//     torch::nn::Conv2d lateral_n2{nullptr}, lateral_n3{nullptr}, lateral_n4{nullptr};
//     torch::nn::Conv2d conv_n2{nullptr}, conv_n3{nullptr}, conv_n4{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(BottomUpPath);
//
// // Adaptive Feature Pooling (Simplified for Semantic Segmentation)
// struct AdaptiveFeaturePoolingImpl : torch::nn::Module {
//     AdaptiveFeaturePoolingImpl() {
//         // 1x1 conv to align channels
//         align_n2 = register_module("align_n2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, 1).stride(1)));
//         align_n3 = register_module("align_n3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, 1).stride(1)));
//         align_n4 = register_module("align_n4", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, 1).stride(1)));
//         align_p5 = register_module("align_p5", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, 1).stride(1)));
//
//         relu = register_module("relu", torch::nn::ReLU());
//     }
//
//     torch::Tensor forward(const std::vector<torch::Tensor>& features) {
//         auto n2 = features[0]; // [batch, 256, 7, 7]
//         auto n3 = features[1]; // [batch, 256, 4, 4]
//         auto n4 = features[2]; // [batch, 256, 2, 2]
//         auto p5 = features[3]; // [batch, 256, 1, 1]
//
//         // Align channels
//         n2 = relu->forward(align_n2->forward(n2)); // [batch, 256, 7, 7]
//         n3 = relu->forward(align_n3->forward(n3)); // [batch, 256, 4, 4]
//         n4 = relu->forward(align_n4->forward(n4)); // [batch, 256, 2, 2]
//         p5 = relu->forward(align_p5->forward(p5)); // [batch, 256, 1, 1]
//
//         // Upsample to common resolution (7x7)
//         n3 = torch::upsample_nearest2d(n3, {7, 7});
//         n4 = torch::upsample_nearest2d(n4, {7, 7});
//         p5 = torch::upsample_nearest2d(p5, {7, 7});
//
//         // Element-wise max fusion
//         auto fused = torch::stack({n2, n3, n4, p5}, 0).max(0).values; // [batch, 256, 7, 7]
//
//         return fused;
//     }
//
//     torch::nn::Conv2d align_n2{nullptr}, align_n3{nullptr}, align_n4{nullptr}, align_p5{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(AdaptiveFeaturePooling);
//
// // Segmentation Head with Complementary Branch
// struct SegmentationHeadImpl : torch::nn::Module {
//     SegmentationHeadImpl(int num_classes) {
//         // FCN branch
//         fcn_branch = register_module("fcn_branch", torch::nn::Sequential(
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 128, 3).stride(1).padding(1)),
//             torch::nn::BatchNorm2d(128),
//             torch::nn::ReLU(),
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(128, num_classes, 1).stride(1))
//         ));
//
//         // FC branch (simplified as conv for semantic segmentation)
//         fc_branch = register_module("fc_branch", torch::nn::Sequential(
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 128, 3).stride(1).padding(1)),
//             torch::nn::BatchNorm2d(128),
//             torch::nn::ReLU(),
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(128, num_classes, 1).stride(1))
//         ));
//
//         // Final upsampling
//         upsample = register_module("upsample", torch::nn::Functional(
//             [](torch::Tensor x) { return torch::upsample_nearest2d(x, {28, 28}, false); }));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // FCN branch
//         auto fcn_out = fcn_branch->forward(x); // [batch, num_classes, 7, 7]
//         fcn_out = upsample->forward(fcn_out); // [batch, num_classes, 28, 28]
//
//         // FC branch
//         auto fc_out = fc_branch->forward(x); // [batch, num_classes, 7, 7]
//         fc_out = upsample->forward(fc_out); // [batch, num_classes, 28, 28]
//
//         // Combine (element-wise addition)
//         return fcn_out + fc_out; // [batch, num_classes, 28, 28]
//     }
//
//     torch::nn::Sequential fcn_branch{nullptr}, fc_branch{nullptr};
//     torch::nn::Functional upsample{nullptr};
// };
// TORCH_MODULE(SegmentationHead);
//
// // PANet Model
// struct PANetImpl : torch::nn::Module {
//     PANetImpl(int num_classes) : num_classes_(num_classes) {
//         backbone = register_module("backbone", FPNBackbone());
//         bottom_up = register_module("bottom_up", BottomUpPath());
//         afp = register_module("afp", AdaptiveFeaturePooling());
//         head = register_module("head", SegmentationHead(num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Backbone with FPN
//         auto fpn_outputs = backbone->forward(x); // {P2, P3, P4, P5, C2, C3, C4, C5}
//
//         // Bottom-up path augmentation
//         auto bottom_up_outputs = bottom_up->forward(
//             {fpn_outputs[0], fpn_outputs[1], fpn_outputs[2], fpn_outputs[3]}); // {N2, N3, N4, P5}
//
//         // Adaptive feature pooling
//         auto fused_features = afp->forward(bottom_up_outputs); // [batch, 256, 7, 7]
//
//         // Segmentation head
//         auto output = head->forward(fused_features); // [batch, num_classes, 28, 28]
//
//         return output;
//     }
//
//     int num_classes_;
//     FPNBackbone backbone{nullptr};
//     BottomUpPath bottom_up{nullptr};
//     AdaptiveFeaturePooling afp{nullptr};
//     SegmentationHead head{nullptr};
// };
// TORCH_MODULE(PANet);
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
//         PANet model(num_classes);
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
//                 cv::imwrite("predicted_panet_epoch_" + std::to_string(epoch + 1) + ".jpg", mask);
//             }
//         }
//
//         // Save model
//         torch::save(model, "panet.pt");
//         std::cout << "Model saved as panet.pt" << std::endl;
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
    PANet::PANet(int num_classes, int in_channels)
    {
    }

    PANet::PANet(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void PANet::reset()
    {
    }

    auto PANet::forward(std::initializer_list<std::any> tensors) -> std::any
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
