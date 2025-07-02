#include "include/models/computer_vision/vision_transformers/pvt.h"


using namespace std;

//PVTV1 GROK


// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // Patch Embedding Layer
// struct PatchEmbedImpl : torch::nn::Module {
//     PatchEmbedImpl(int img_size, int patch_size, int in_channels, int embed_dim) {
//         num_patches_ = (img_size / patch_size) * (img_size / patch_size);
//         proj = register_module("proj", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, embed_dim, patch_size).stride(patch_size)));
//         norm = register_module("norm", torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({embed_dim}).eps(1e-6)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, in_channels, img_size, img_size]
//         x = proj->forward(x); // [batch, embed_dim, img_size/patch_size, img_size/patch_size]
//         x = x.flatten(2).transpose(1, 2); // [batch, num_patches, embed_dim]
//         x = norm->forward(x);
//         return x;
//     }
//
//     int num_patches_;
//     torch::nn::Conv2d proj{nullptr};
//     torch::nn::LayerNorm norm{nullptr};
// };
// TORCH_MODULE(PatchEmbed);
//
// // Spatial Reduction Attention (SRA)
// struct SRAImpl : torch::nn::Module {
//     SRAImpl(int embed_dim, int num_heads, int reduction_ratio) {
//         num_heads_ = num_heads;
//         head_dim_ = embed_dim / num_heads;
//         scale_ = 1.0 / std::sqrt(head_dim_);
//         reduction_ratio_ = reduction_ratio;
//         q = register_module("q", torch::nn::Linear(embed_dim, embed_dim));
//         kv = register_module("kv", torch::nn::Linear(embed_dim, embed_dim * 2));
//         proj = register_module("proj", torch::nn::Linear(embed_dim, embed_dim));
//         if (reduction_ratio > 1) {
//             sr = register_module("sr", torch::nn::Conv2d(
//                 torch::nn::Conv2dOptions(embed_dim, embed_dim, reduction_ratio).stride(reduction_ratio)));
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, seq_len, embed_dim]
//         auto sizes = x.sizes();
//         int batch = sizes[0], seq_len = sizes[1];
//         int h = static_cast<int>(std::sqrt(seq_len)); // Assume square feature map
//
//         // Query
//         auto q_out = q->forward(x); // [batch, seq_len, embed_dim]
//         q_out = q_out.view({batch, seq_len, num_heads_, head_dim_})
//                      .permute({0, 2, 1, 3}); // [batch, num_heads, seq_len, head_dim]
//
//         // Key and Value with Spatial Reduction
//         auto kv_out = kv->forward(x); // [batch, seq_len, embed_dim * 2]
//         kv_out = kv_out.view({batch, seq_len, 2, num_heads_, head_dim_})
//                        .permute({2, 0, 3, 1, 4}); // [2, batch, num_heads, seq_len, head_dim]
//         auto k = kv_out[0], v = kv_out[1]; // [batch, num_heads, seq_len, head_dim]
//
//         if (reduction_ratio_ > 1) {
//             // Reshape for spatial reduction
//             auto x_reshaped = x.transpose(1, 2).view({batch, -1, h, h}); // [batch, embed_dim, h, h]
//             x_reshaped = sr->forward(x_reshaped); // [batch, embed_dim, h/reduction_ratio, h/reduction_ratio]
//             int reduced_seq_len = x_reshaped.size(2) * x_reshaped.size(3);
//             x_reshaped = x_reshaped.flatten(2).transpose(1, 2); // [batch, reduced_seq_len, embed_dim]
//             kv_out = kv->forward(x_reshaped).view({batch, reduced_seq_len, 2, num_heads_, head_dim_})
//                                             .permute({2, 0, 3, 1, 4}); // [2, batch, num_heads, reduced_seq_len, head_dim]
//             k = kv_out[0]; v = kv_out[1]; // [batch, num_heads, reduced_seq_len, head_dim]
//         }
//
//         // Attention
//         auto attn = torch::matmul(q_out, k.transpose(-2, -1)) * scale_; // [batch, num_heads, seq_len, reduced_seq_len]
//         attn = torch::softmax(attn, -1);
//         auto out = torch::matmul(attn, v); // [batch, num_heads, seq_len, head_dim]
//         out = out.permute({0, 2, 1, 3}).contiguous()
//                  .view({batch, seq_len, num_heads_ * head_dim_}); // [batch, seq_len, embed_dim]
//         out = proj->forward(out);
//         return out;
//     }
//
//     int num_heads_, head_dim_, reduction_ratio_;
//     float scale_;
//     torch::nn::Linear q{nullptr}, kv{nullptr}, proj{nullptr};
//     torch::nn::Conv2d sr{nullptr};
// };
// TORCH_MODULE(SRA);
//
// // Transformer Block with SRA
// struct TransformerBlockImpl : torch::nn::Module {
//     TransformerBlockImpl(int embed_dim, int num_heads, int reduction_ratio, int mlp_ratio = 4, float dropout = 0.1) {
//         attn = register_module("attn", SRA(embed_dim, num_heads, reduction_ratio));
//         norm1 = register_module("norm1", torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({embed_dim}).eps(1e-6)));
//         mlp = register_module("mlp", torch::nn::Sequential(
//             torch::nn::Linear(embed_dim, embed_dim * mlp_ratio),
//             torch::nn::GELU(),
//             torch::nn::Dropout(dropout),
//             torch::nn::Linear(embed_dim * mlp_ratio, embed_dim),
//             torch::nn::Dropout(dropout)));
//         norm2 = register_module("norm2", torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({embed_dim}).eps(1e-6)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto residual = x;
//         x = norm1->forward(x);
//         x = attn->forward(x);
//         x = residual + x;
//         residual = x;
//         x = norm2->forward(x);
//         x = mlp->forward(x);
//         x = residual + x;
//         return x;
//     }
//
//     SRA attn{nullptr};
//     torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
//     torch::nn::Sequential mlp{nullptr};
// };
// TORCH_MODULE(TransformerBlock);
//
// // PVT v1 Model
// struct PVTImpl : torch::nn::Module {
//     PVTImpl(int img_size, int in_channels, int num_classes) {
//         // Stage 1: embed_dim=64, patch_size=7, seq_len=16
//         stage1_embed = register_module("stage1_embed", PatchEmbed(img_size, 7, in_channels, 64));
//         for (int i = 0; i < 2; ++i) {
//             stage1->push_back(TransformerBlock(64, 1, 8));
//             register_module("stage1_block_" + std::to_string(i), stage1[i]);
//         }
//
//         // Stage 2: embed_dim=128, patch_size=2, seq_len=4
//         stage2_embed = register_module("stage2_embed", PatchEmbed(img_size / 7, 2, 64, 128));
//         for (int i = 0; i < 2; ++i) {
//             stage2->push_back(TransformerBlock(128, 2, 4));
//             register_module("stage2_block_" + std::to_string(i), stage2[i]);
//         }
//
//         // Head
//         norm = register_module("norm", torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({128}).eps(1e-6)));
//         head = register_module("head", torch::nn::Linear(128, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, in_channels, img_size, img_size]
//
//         // Stage 1
//         x = stage1_embed->forward(x); // [batch, 16, 64]
//         for (auto& block : *stage1) {
//             x = block->forward(x); // [batch, 16, 64]
//         }
//
//         // Stage 2
//         // Reshape for next patch embedding
//         int batch = x.size(0), seq_len = x.size(1);
//         int h = static_cast<int>(std::sqrt(seq_len)); // 4
//         x = x.transpose(1, 2).view({batch, 64, h, h}); // [batch, 64, 4, 4]
//         x = stage2_embed->forward(x); // [batch, 4, 128]
//         for (auto& block : *stage2) {
//             x = block->forward(x); // [batch, 4, 128]
//         }
//
//         // Pool and classify
//         x = norm->forward(x);
//         x = x.mean(1); // [batch, 128]
//         x = head->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     PatchEmbed stage1_embed{nullptr}, stage2_embed{nullptr};
//     torch::nn::ModuleList stage1{nullptr}, stage2{nullptr};
//     torch::nn::LayerNorm norm{nullptr};
//     torch::nn::Linear head{nullptr};
// };
// TORCH_MODULE(PVT);
//
// // Dataset for Images and Labels
// struct ImageDataset : torch::data::Dataset<ImageDataset> {
//     ImageDataset(const std::string& img_dir, const std::string& label_dir) {
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
//         const int img_size = 28;
//         const int in_channels = 1;
//         const int num_classes = 10; // e.g., MNIST digits
//         const int batch_size = 16;
//         const float lr = 0.001;
//         const int num_epochs = 20;
//
//         // Initialize model
//         PVT model(img_size, in_channels, num_classes);
//         model->to(device);
//
//         // Optimizer
//         torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));
//
//         // Loss function
//         auto ce_loss = torch::nn::CrossEntropyLoss();
//
//         // Load dataset
//         auto dataset = ImageDataset("./data/images", "./data/labels")
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
//                 auto logits = model->forward(images); // [batch, num_classes]
//                 auto loss = ce_loss->forward(logits, labels);
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
//             // Save model every 10 epochs
//             if ((epoch + 1) % 10 == 0) {
//                 torch::save(model, "pvt_epoch_" + std::to_string(epoch + 1) + ".pt");
//             }
//         }
//
//         // Save final model
//         torch::save(model, "pvt.pt");
//         std::cout << "Model saved as pvt.pt" << std::endl;
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
//
//     return 0;
// }


//PVTV2 GROK

//
// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // Overlapping Patch Embedding Layer
// struct PatchEmbedImpl : torch::nn::Module {
//     PatchEmbedImpl(int patch_size, int stride, int in_channels, int embed_dim) {
//         proj = register_module("proj", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, embed_dim, patch_size).stride(stride).padding(patch_size / 2)));
//         norm = register_module("norm", torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({embed_dim}).eps(1e-6)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, in_channels, h, w]
//         x = proj->forward(x); // [batch, embed_dim, h', w']
//         x = x.flatten(2).transpose(1, 2); // [batch, num_patches, embed_dim]
//         x = norm->forward(x);
//         return x;
//     }
//
//     torch::nn::Conv2d proj{nullptr};
//     torch::nn::LayerNorm norm{nullptr};
// };
// TORCH_MODULE(PatchEmbed);
//
// // Linear Spatial Reduction Attention (SRA)
// struct LinearSRAImpl : torch::nn::Module {
//     LinearSRAImpl(int embed_dim, int num_heads, int reduction_ratio) {
//         num_heads_ = num_heads;
//         head_dim_ = embed_dim / num_heads;
//         scale_ = 1.0 / std::sqrt(head_dim_);
//         reduction_ratio_ = reduction_ratio;
//         qkv = register_module("qkv", torch::nn::Linear(embed_dim, embed_dim * 3));
//         proj = register_module("proj", torch::nn::Linear(embed_dim, embed_dim));
//         if (reduction_ratio > 1) {
//             sr = register_module("sr", torch::nn::Linear(embed_dim, embed_dim / reduction_ratio));
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, seq_len, embed_dim]
//         auto sizes = x.sizes();
//         int batch = sizes[0], seq_len = sizes[1];
//
//         // QKV projection
//         auto qkv_out = qkv->forward(x); // [batch, seq_len, embed_dim * 3]
//         qkv_out = qkv_out.view({batch, seq_len, 3, num_heads_, head_dim_})
//                          .permute({2, 0, 3, 1, 4}); // [3, batch, num_heads, seq_len, head_dim]
//         auto q = qkv_out[0], k = qkv_out[1], v = qkv_out[2]; // [batch, num_heads, seq_len, head_dim]
//
//         // Linear spatial reduction for key and value
//         if (reduction_ratio_ > 1) {
//             auto x_reshaped = x; // [batch, seq_len, embed_dim]
//             x_reshaped = sr->forward(x_reshaped); // [batch, seq_len, embed_dim / reduction_ratio]
//             auto kv_out = qkv->forward(x_reshaped).view({batch, seq_len, 3, num_heads_, head_dim_ / reduction_ratio_})
//                                                   .permute({2, 0, 3, 1, 4}); // [3, batch, num_heads, seq_len, head_dim/reduction_ratio]
//             k = kv_out[1]; v = kv_out[2]; // [batch, num_heads, seq_len, head_dim/reduction_ratio]
//         }
//
//         // Attention
//         auto attn = torch::matmul(q, k.transpose(-2, -1)) * scale_; // [batch, num_heads, seq_len, seq_len]
//         attn = torch::softmax(attn, -1);
//         auto out = torch::matmul(attn, v); // [batch, num_heads, seq_len, head_dim]
//         out = out.permute({0, 2, 1, 3}).contiguous()
//                  .view({batch, seq_len, num_heads_ * head_dim_}); // [batch, seq_len, embed_dim]
//         out = proj->forward(out);
//         return out;
//     }
//
//     int num_heads_, head_dim_, reduction_ratio_;
//     float scale_;
//     torch::nn::Linear qkv{nullptr}, proj{nullptr}, sr{nullptr};
// };
// TORCH_MODULE(LinearSRA);
//
// // Convolutional Feed-Forward Network (CFFN)
// struct CFFNImpl : torch::nn::Module {
//     CFFNImpl(int embed_dim, int mlp_ratio = 4, float dropout = 0.1) {
//         fc1 = register_module("fc1", torch::nn::Linear(embed_dim, embed_dim * mlp_ratio));
//         conv = register_module("conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(embed_dim * mlp_ratio, embed_dim * mlp_ratio, 3).padding(1).groups(embed_dim * mlp_ratio)));
//         fc2 = register_module("fc2", torch::nn::Linear(embed_dim * mlp_ratio, embed_dim));
//         dropout1 = register_module("dropout1", torch::nn::Dropout(dropout));
//         dropout2 = register_module("dropout2", torch::nn::Dropout(dropout));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, seq_len, embed_dim]
//         auto sizes = x.sizes();
//         int batch = sizes[0], seq_len = sizes[1];
//         int h = static_cast<int>(std::sqrt(seq_len)); // Assume square feature map
//
//         x = fc1->forward(x); // [batch, seq_len, embed_dim * mlp_ratio]
//         x = dropout1->forward(x);
//         x = x.transpose(1, 2).view({batch, -1, h, h}); // [batch, embed_dim * mlp_ratio, h, h]
//         x = conv->forward(x); // [batch, embed_dim * mlp_ratio, h, h]
//         x = x.flatten(2).transpose(1, 2); // [batch, seq_len, embed_dim * mlp_ratio]
//         x = torch::gelu(x);
//         x = fc2->forward(x); // [batch, seq_len, embed_dim]
//         x = dropout2->forward(x);
//         return x;
//     }
//
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr};
//     torch::nn::Conv2d conv{nullptr};
//     torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr};
// };
// TORCH_MODULE(CFFN);
//
// // Transformer Block with Linear SRA and CFFN
// struct TransformerBlockImpl : torch::nn::Module {
//     TransformerBlockImpl(int embed_dim, int num_heads, int reduction_ratio, int mlp_ratio = 4, float dropout = 0.1) {
//         attn = register_module("attn", LinearSRA(embed_dim, num_heads, reduction_ratio));
//         norm1 = register_module("norm1", torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({embed_dim}).eps(1e-6)));
//         ffn = register_module("ffn", CFFN(embed_dim, mlp_ratio, dropout));
//         norm2 = register_module("norm2", torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({embed_dim}).eps(1e-6)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto residual = x;
//         x = norm1->forward(x);
//         x = attn->forward(x);
//         x = residual + x;
//         residual = x;
//         x = norm2->forward(x);
//         x = ffn->forward(x);
//         x = residual + x;
//         return x;
//     }
//
//     LinearSRA attn{nullptr};
//     torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
//     CFFN ffn{nullptr};
// };
// TORCH_MODULE(TransformerBlock);
//
// // PVT v2 Model
// struct PVTv2Impl : torch::nn::Module {
//     PVTv2Impl(int img_size, int in_channels, int num_classes) {
//         // Stage 1: embed_dim=64, patch_size=7, stride=4, seq_len=49
//         stage1_embed = register_module("stage1_embed", PatchEmbed(7, 4, in_channels, 64));
//         for (int i = 0; i < 2; ++i) {
//             stage1->push_back(TransformerBlock(64, 1, 8));
//             register_module("stage1_block_" + std::to_string(i), stage1[i]);
//         }
//
//         // Stage 2: embed_dim=128, patch_size=3, stride=2, seq_len=16
//         stage2_embed = register_module("stage2_embed", PatchEmbed(3, 2, 64, 128));
//         for (int i = 0; i < 2; ++i) {
//             stage2->push_back(TransformerBlock(128, 2, 4));
//             register_module("stage2_block_" + std::to_string(i), stage2[i]);
//         }
//
//         // Head
//         norm = register_module("norm", torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({128}).eps(1e-6)));
//         head = register_module("head", torch::nn::Linear(128, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, in_channels, img_size, img_size]
//
//         // Stage 1
//         x = stage1_embed->forward(x); // [batch, 49, 64] (28/4)^2
//         for (auto& block : *stage1) {
//             x = block->forward(x); // [batch, 49, 64]
//         }
//
//         // Stage 2
//         // Reshape for next patch embedding
//         int batch = x.size(0), seq_len = x.size(1);
//         int h = static_cast<int>(std::sqrt(seq_len)); // 7
//         x = x.transpose(1, 2).view({batch, 64, h, h}); // [batch, 64, 7, 7]
//         x = stage2_embed->forward(x); // [batch, 16, 128] (7/2)^2
//         for (auto& block : *stage2) {
//             x = block->forward(x); // [batch, 16, 128]
//         }
//
//         // Pool and classify
//         x = norm->forward(x);
//         x = x.mean(1); // [batch, 128]
//         x = head->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     PatchEmbed stage1_embed{nullptr}, stage2_embed{nullptr};
//     torch::nn::ModuleList stage1{nullptr}, stage2{nullptr};
//     torch::nn::LayerNorm norm{nullptr};
//     torch::nn::Linear head{nullptr};
// };
// TORCH_MODULE(PVTv2);
//
// // Dataset for Images and Labels
// struct ImageDataset : torch::data::Dataset<ImageDataset> {
//     ImageDataset(const std::string& img_dir, const std::string& label_dir) {
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
//         const int img_size = 28;
//         const int in_channels = 1;
//         const int num_classes = 10; // e.g., MNIST digits
//         const int batch_size = 16;
//         const float lr = 0.001;
//         const int num_epochs = 20;
//
//         // Initialize model
//         PVTv2 model(img_size, in_channels, num_classes);
//         model->to(device);
//
//         // Optimizer
//         torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));
//
//         // Loss function
//         auto ce_loss = torch::nn::CrossEntropyLoss();
//
//         // Load dataset
//         auto dataset = ImageDataset("./data/images", "./data/labels")
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
//                 auto logits = model->forward(images); // [batch, num_classes]
//                 auto loss = ce_loss->forward(logits, labels);
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
//             // Save model every 10 epochs
//             if ((epoch + 1) % 10 == 0) {
//                 torch::save(model, "pvt_v2_epoch_" + std::to_string(epoch + 1) + ".pt");
//             }
//         }
//
//         // Save final model
//         torch::save(model, "pvt_v2.pt");
//         std::cout << "Model saved as pvt_v2.pt" << std::endl;
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
    PVTV1::PVTV1(int num_classes, int in_channels)
    {
    }

    PVTV1::PVTV1(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void PVTV1::reset()
    {
    }

    auto PVTV1::forward(std::initializer_list<std::any> tensors) -> std::any
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



    PVTV2::PVTV2(int num_classes, int in_channels)
    {
    }

    PVTV2::PVTV2(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void PVTV2::reset()
    {
    }

    auto PVTV2::forward(std::initializer_list<std::any> tensors) -> std::any
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
