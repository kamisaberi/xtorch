#include "include/models/computer_vision/image_segmentation/hrnet.h"


using namespace std;
//HRNet GROK

// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // Basic Block for HRNet (BN + ReLU + Conv)
// struct BasicBlockImpl : torch::nn::Module {
//     BasicBlockImpl(int in_channels, int out_channels, int stride = 1) {
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));
//         relu = register_module("relu", torch::nn::ReLU());
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(1)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
//
//         if (stride != 1 || in_channels != out_channels) {
//             shortcut = register_module("shortcut", torch::nn::Sequential(
//                 torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride)),
//                 torch::nn::BatchNorm2d(out_channels)
//             ));
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto residual = x;
//         x = relu->forward(bn1->forward(conv1->forward(x)));
//         x = bn2->forward(conv2->forward(x));
//
//         if (shortcut.defined()) {
//             residual = shortcut->forward(residual);
//         }
//
//         x = relu->forward(x + residual);
//         return x;
//     }
//
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
//     torch::nn::Sequential shortcut{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(BasicBlock);
//
// // Stage Module for HRNet (Parallel Streams with Fusion)
// struct StageModuleImpl : torch::nn::Module {
//     StageModuleImpl(const std::vector<int>& channels, int num_blocks, bool add_stream = false) {
//         // Initialize parallel streams
//         for (size_t i = 0; i < channels.size(); ++i) {
//             torch::nn::Sequential stream;
//             for (int j = 0; j < num_blocks; ++j) {
//                 stream->push_back("block_" + std::to_string(j), BasicBlock(
//                     channels[i], channels[i], i == channels.size() - 1 && add_stream ? 2 : 1));
//             }
//             streams->push_back("stream_" + std::to_string(i), stream);
//         }
//
//         // Fusion layers (if multi-resolution)
//         if (channels.size() > 1 || add_stream) {
//             for (size_t i = 0; i < channels.size() + (add_stream ? 1 : 0); ++i) {
//                 torch::nn::SequentialDict fusion;
//                 for (size_t j = 0; j < channels.size() + (add_stream ? 1 : 0); ++j) {
//                     if (i != j) {
//                         auto in_ch = i < channels.size() ? channels[i] : channels.back() * 2;
//                         auto out_ch = j < channels.size() ? channels[j] : channels.back() * 2;
//                         torch::nn::Sequential layer;
//                         if (j > i) {
//                             // Downsample
//                             layer->push_back("conv", torch::nn::Conv2d(
//                                 torch::nn::Conv2dOptions(in_ch, out_ch, 3).stride(2).padding(1)));
//                             layer->push_back("bn", torch::nn::BatchNorm2d(out_ch));
//                             layer->push_back("relu", torch::nn::ReLU());
//                         } else {
//                             // Upsample or identity
//                             if (j < i) {
//                                 layer->push_back("upsample", torch::nn::Functional(
//                                     [out_h = 28 / (1 << j), out_w = 28 / (1 << j)](torch::Tensor x) {
//                                         return torch::upsample_bilinear2d(x, {out_h, out_w}, true);
//                                     }));
//                             }
//                             layer->push_back("conv", torch::nn::Conv2d(
//                                 torch::nn::Conv2dOptions(in_ch, out_ch, 1).stride(1)));
//                             layer->push_back("bn", torch::nn::BatchNorm2d(out_ch));
//                         }
//                         fusion->insert("to_" + std::to_string(j), layer);
//                     }
//                 }
//                 fuse_layers->push_back("fuse_" + std::to_string(i), fusion);
//             }
//         }
//
//         streams = register_module("streams", streams);
//         fuse_layers = register_module("fuse_layers", fuse_layers);
//         relu = register_module("relu", torch::nn::ReLU());
//     }
//
//     std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) {
//         // Process each stream
//         std::vector<torch::Tensor> outputs;
//         for (size_t i = 0; i < streams->size(); ++i) {
//             outputs.push_back(streams[i]->forward(inputs[i]));
//         }
//
//         // Multi-scale fusion
//         if (!fuse_layers->empty()) {
//             std::vector<torch::Tensor> fused_outputs(outputs.size(), torch::zeros_like(outputs[0]));
//             for (size_t i = 0; i < outputs.size(); ++i) {
//                 for (size_t j = 0; j < outputs.size(); ++j) {
//                     if (i == j) {
//                         fused_outputs[j] = fused_outputs[j] + outputs[j];
//                     } else {
//                         auto layer = fuse_layers[i]->second->get("to_" + std::to_string(j));
//                         fused_outputs[j] = fused_outputs[j] + layer->forward(outputs[i]);
//                     }
//                 }
//             }
//             for (auto& out : fused_outputs) {
//                 out = relu->forward(out);
//             }
//             return fused_outputs;
//         }
//
//         return outputs;
//     }
//
//     torch::nn::SequentialList streams{nullptr};
//     torch::nn::SequentialDictList fuse_layers{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(StageModule);
//
// // HRNet Model
// struct HRNetImpl : torch::nn::Module {
//     HRNetImpl(int num_classes) : num_classes_(num_classes) {
//         // Initial convolution
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(1, 64, 3).stride(2).padding(1))); // [batch, 64, 14, 14]
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
//         relu = register_module("relu", torch::nn::ReLU());
//
//         // Stage 1: Single high-resolution stream
//         stage1 = register_module("stage1", StageModule({64}, {2}));
//
//         // Stage 2: Add low-resolution stream
//         stage2 = register_module("stage2", StageModule({32, 16}, {2}, true));
//
//         // Stage 3: Maintain two streams
//         stage3 = register_module("stage3", StageModule({32, 64}, {2}));
//
//         // Segmentation head
//         head = register_module("head", torch::nn::Sequential(
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3).stride(1).padding(1)),
//             torch::nn::BatchNorm2d(32),
//             torch::nn::ReLU(),
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(32, num_classes, 1).stride(1)),
//             torch::nn::Functional([](torch::Tensor x) {
//                 return torch::upsample_bilinear2d(x, {28, 28}, false);
//             })
//         ));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Initial conv
//         x = relu->forward(bn1->forward(conv1->forward(x))); // [batch, 64, 14, 14]
//
//         // Stage 1
//         auto stage1_out = stage1->forward({x}); // [batch, 64, 14, 14]
//
//         // Stage 2
//         auto stage2_out = stage2->forward(stage1_out); // [[batch, 32, 14, 14], [batch, 16, 7, 7]]
//
//         // Stage 3
//         auto stage3_out = stage3->forward(stage2_out); // [[batch, 32, 14, 14], [batch, 64, 7, 7]]
//
//         // Segmentation head (use high-resolution branch)
//         x = head->forward(stage3_out[0]); // [batch, num_classes, 28, 28]
//
//         return x;
//     }
//
//     int num_classes_;
//     torch::nn::Conv2d conv1{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr};
//     torch::nn::ReLU relu{nullptr};
//     StageModule stage1{nullptr}, stage2{nullptr}, stage3{nullptr};
//     torch::nn::Sequential head{nullptr};
// };
// TORCH_MODULE(HRNet);
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
//         HRNet model(num_classes);
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
//                 cv::imwrite("predicted_hrnet_epoch_" + std::to_string(epoch + 1) + ".jpg", mask);
//             }
//         }
//
//         // Save model
//         torch::save(model, "hrnet.pt");
//         std::cout << "Model saved as hrnet.pt" << std::endl;
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
    HRNet::HRNet(int num_classes, int in_channels)
    {
    }

    HRNet::HRNet(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void HRNet::reset()
    {
    }

    auto HRNet::forward(std::initializer_list<std::any> tensors) -> std::any
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
