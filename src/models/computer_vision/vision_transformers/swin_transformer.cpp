#include "include/models/computer_vision/vision_transformers/swin_transformer.h"


using namespace std;


//SwinTransformerV1 GROK



// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
// #include <cmath>
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
// // Window Partition and Reverse Functions
// std::tuple<torch::Tensor, std::vector<int64_t>> window_partition(torch::Tensor x, int window_size) {
//     // x: [batch, num_patches, embed_dim], where num_patches = h * w
//     auto sizes = x.sizes();
//     int batch = sizes[0], embed_dim = sizes[2];
//     int h = static_cast<int>(std::sqrt(sizes[1])), w = h; // Assume square
//     x = x.view({batch, h, w, embed_dim}); // [batch, h, w, embed_dim]
//     x = x.view({batch, h / window_size, window_size, w / window_size, window_size, embed_dim})
//          .permute({0, 1, 3, 2, 4, 5}) // [batch, h/window_size, w/window_size, window_size, window_size, embed_dim]
//          .contiguous()
//          .view({-1, window_size * window_size, embed_dim}); // [num_windows*batch, window_size*window_size, embed_dim]
//     return {x, {batch, h / window_size, w / window_size}};
// }
//
// torch::Tensor window_reverse(torch::Tensor windows, int window_size, const std::vector<int64_t>& shape) {
//     // windows: [num_windows*batch, window_size*window_size, embed_dim]
//     int batch = shape[0], h_win = shape[1], w_win = shape[2];
//     windows = windows.view({batch, h_win, w_win, window_size, window_size, -1}) // [batch, h_win, w_win, window_size, window_size, embed_dim]
//                      .permute({0, 1, 3, 2, 4, 5}) // [batch, h_win, window_size, w_win, window_size, embed_dim]
//                      .contiguous()
//                      .view({batch, h_win * window_size, w_win * window_size, -1}); // [batch, h, w, embed_dim]
//     return windows.view({batch, -1, windows.size(-1)}); // [batch, num_patches, embed_dim]
// }
//
// // Window-based Multi-Head Self-Attention (W-MSA)
// struct WMSAImpl : torch::nn::Module {
//     WMSAImpl(int embed_dim, int num_heads, int window_size, bool shift = false) : window_size_(window_size), shift_(shift) {
//         num_heads_ = num_heads;
//         head_dim_ = embed_dim / num_heads;
//         scale_ = 1.0 / std::sqrt(head_dim_);
//         qkv = register_module("qkv", torch::nn::Linear(embed_dim, embed_dim * 3));
//         proj = register_module("proj", torch::nn::Linear(embed_dim, embed_dim));
//         if (shift) {
//             // Relative positional bias table
//             relative_position_bias_table = register_parameter("relative_position_bias_table",
//                 torch::randn({(2 * window_size - 1) * (2 * window_size - 1), num_heads}));
//             // Index for relative positions
//             auto coords_h = torch::arange(window_size);
//             auto coords_w = torch::arange(window_size);
//             auto coords = torch::stack(torch::meshgrid({coords_h, coords_w}, false)); // [2, window_size, window_size]
//             coords = coords.flatten(1).transpose(0, 1); // [window_size*window_size, 2]
//             relative_coords = coords.unsqueeze(1) - coords.unsqueeze(0); // [window_size*window_size, window_size*window_size, 2]
//             relative_coords = relative_coords.view({-1, 2}).to(torch::kInt64); // [window_size*window_size*window_size*window_size, 2]
//             relative_coords.index_add_(1, torch::tensor({0}), torch::tensor({window_size - 1}));
//             relative_coords.index_add_(1, torch::tensor({1}), torch::tensor({2 * window_size - 1}));
//             relative_position_index = register_buffer("relative_position_index", relative_coords.flatten());
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, num_patches, embed_dim]
//         auto sizes = x.sizes();
//         int batch = sizes[0], num_patches = sizes[1];
//         int h = static_cast<int>(std::sqrt(num_patches)), w = h;
//
//         // Shift window if needed
//         torch::Tensor x_shifted = x;
//         if (shift_) {
//             x_shifted = torch::roll(x.view({batch, h, w, -1}), {-window_size_ / 2, -window_size_ / 2}, {1, 2})
//                            .view({batch, num_patches, -1});
//         }
//
//         // Window partition
//         auto [windows, window_shape] = window_partition(x_shifted, window_size_); // [num_windows*batch, window_size*window_size, embed_dim]
//
//         // QKV projection
//         auto qkv_out = qkv->forward(windows); // [num_windows*batch, window_size*window_size, embed_dim*3]
//         qkv_out = qkv_out.view({-1, window_size_ * window_size_, 3, num_heads_, head_dim_})
//                          .permute({2, 0, 3, 1, 4}); // [3, num_windows*batch, num_heads, window_size*window_size, head_dim]
//         auto q = qkv_out[0], k = qkv_out[1], v = qkv_out[2]; // [num_windows*batch, num_heads, window_size*window_size, head_dim]
//
//         // Attention
//         auto attn = torch::matmul(q, k.transpose(-2, -1)) * scale_; // [num_windows*batch, num_heads, window_size*window_size, window_size*window_size]
//
//         // Add relative positional bias if needed
//         if (shift_) {
//             auto bias = relative_position_bias_table.index_select(0, relative_position_index); // [window_size*window_size*window_size*window_size, num_heads]
//             bias = bias.view({window_size_ * window_size_, window_size_ * window_size_, num_heads_})
//                        .permute({2, 0, 1}); // [num_heads, window_size*window_size, window_size*window_size]
//             attn = attn + bias.unsqueeze(0); // [num_windows*batch, num_heads, window_size*window_size, window_size*window_size]
//         }
//
//         attn = torch::softmax(attn, -1);
//         auto out = torch::matmul(attn, v); // [num_windows*batch, num_heads, window_size*window_size, head_dim]
//         out = out.permute({0, 2, 1, 3}).contiguous()
//                  .view({-1, window_size_ * window_size_, num_heads_ * head_dim_}); // [num_windows*batch, window_size*window_size, embed_dim]
//         out = proj->forward(out); // [num_windows*batch, window_size*window_size, embed_dim]
//
//         // Reverse window partition
//         out = window_reverse(out, window_size_, window_shape); // [batch, num_patches, embed_dim]
//
//         // Reverse shift if needed
//         if (shift_) {
//             out = torch::roll(out.view({batch, h, w, -1}), {window_size_ / 2, window_size_ / 2}, {1, 2})
//                      .view({batch, num_patches, -1});
//         }
//
//         return out;
//     }
//
//     int num_heads_, head_dim_, window_size_;
//     bool shift_;
//     float scale_;
//     torch::nn::Linear qkv{nullptr}, proj{nullptr};
//     torch::Tensor relative_position_bias_table, relative_position_index;
// };
// TORCH_MODULE(WMSA);
//
// // Swin Transformer Block
// struct SwinBlockImpl : torch::nn::Module {
//     SwinBlockImpl(int embed_dim, int num_heads, int window_size, bool shift = false, int mlp_ratio = 4, float dropout = 0.1) {
//         norm1 = register_module("norm1", torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({embed_dim}).eps(1e-6)));
//         attn = register_module("attn", WMSA(embed_dim, num_heads, window_size, shift));
//         norm2 = register_module("norm2", torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({embed_dim}).eps(1e-6)));
//         mlp = register_module("mlp", torch::nn::Sequential(
//             torch::nn::Linear(embed_dim, embed_dim * mlp_ratio),
//             torch::nn::GELU(),
//             torch::nn::Dropout(dropout),
//             torch::nn::Linear(embed_dim * mlp_ratio, embed_dim),
//             torch::nn::Dropout(dropout)));
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
//     torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
//     WMSA attn{nullptr};
//     torch::nn::Sequential mlp{nullptr};
// };
// TORCH_MODULE(SwinBlock);
//
// // Swin Transformer v1 Model
// struct SwinTransformerV1Impl : torch::nn::Module {
//     SwinTransformerV1Impl(int img_size, int patch_size, int in_channels, int num_classes, int window_size = 4) {
//         patch_embed = register_module("patch_embed", PatchEmbed(img_size, patch_size, in_channels, 96));
//
//         // Stage 1: embed_dim=96, num_patches=16
//         stage1->push_back(SwinBlock(96, 3, window_size, false));
//         stage1->push_back(SwinBlock(96, 3, window_size, true));
//         register_module("stage1_block_0", stage1[0]);
//         register_module("stage1_block_1", stage1[1]);
//
//         // Stage 2: embed_dim=192, num_patches=4
//         patch_merge1 = register_module("patch_merge1", torch::nn::Linear(96 * 4, 192));
//         stage2->push_back(SwinBlock(192, 6, window_size, false));
//         stage2->push_back(SwinBlock(192, 6, window_size, true));
//         register_module("stage2_block_0", stage2[0]);
//         register_module("stage2_block_1", stage2[1]);
//
//         // Head
//         norm = register_module("norm", torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({192}).eps(1e-6)));
//         head = register_module("head", torch::nn::Linear(192, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, in_channels, img_size, img_size]
//         x = patch_embed->forward(x); // [batch, 16, 96]
//
//         // Stage 1
//         for (auto& block : *stage1) {
//             x = block->forward(x); // [batch, 16, 96]
//         }
//
//         // Patch merging
//         int batch = x.size(0), seq_len = x.size(1);
//         int h = static_cast<int>(std::sqrt(seq_len)); // 4
//         x = x.view({batch, h, h, -1}) // [batch, 4, 4, 96]
//              .transpose(2, 3) // [batch, 4, 96, 4]
//              .contiguous()
//              .view({batch, h, -1}); // [batch, 4, 96*4]
//         x = x.view({batch, h / 2, 2, -1}) // [batch, 2, 2, 96*4]
//              .transpose(2, 3) // [batch, 2, 96*4, 2]
//              .contiguous()
//              .view({batch, -1, 96 * 4}); // [batch, 4, 96*4]
//         x = patch_merge1->forward(x); // [batch, 4, 192]
//
//         // Stage 2
//         for (auto& block : *stage2) {
//             x = block->forward(x); // [batch, 4, 192]
//         }
//
//         // Pool and classify
//         x = norm->forward(x);
//         x = x.mean(1); // [batch, 192]
//         x = head->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     PatchEmbed patch_embed{nullptr};
//     torch::nn::ModuleList stage1{nullptr}, stage2{nullptr};
//     torch::nn::Linear patch_merge1{nullptr}, head{nullptr};
//     torch::nn::LayerNorm norm{nullptr};
// };
// TORCH_MODULE(SwinTransformerV1);
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
//         const int patch_size = 7;
//         const int in_channels = 1;
//         const int num_classes = 10; // e.g., MNIST digits
//         const int window_size = 4; // Adjusted for small input
//         const int batch_size = 16;
//         const float lr = 0.001;
//         const int num_epochs = 20;
//
//         // Initialize model
//         SwinTransformerV1 model(img_size, patch_size, in_channels, num_classes, window_size);
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
//                 torch::save(model, "swin_v1_epoch_" + std::to_string(epoch + 1) + ".pt");
//             }
//         }
//
//         // Save final model
//         torch::save(model, "swin_v1.pt");
//         std::cout << "Model saved as swin_v1.pt" << std::endl;
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
//
//     return 0;
// }


//SwinTransformerV2 GROK


// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
// #include <cmath>
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
// // Window Partition and Reverse Functions
// std::tuple<torch::Tensor, std::vector<int64_t>> window_partition(torch::Tensor x, int window_size) {
//     // x: [batch, num_patches, embed_dim], where num_patches = h * w
//     auto sizes = x.sizes();
//     int batch = sizes[0], embed_dim = sizes[2];
//     int h = static_cast<int>(std::sqrt(sizes[1])), w = h; // Assume square
//     x = x.view({batch, h, w, embed_dim}); // [batch, h, w, embed_dim]
//     x = x.view({batch, h / window_size, window_size, w / window_size, window_size, embed_dim})
//          .permute({0, 1, 3, 2, 4, 5}) // [batch, h/window_size, w/window_size, window_size, window_size, embed_dim]
//          .contiguous()
//          .view({-1, window_size * window_size, embed_dim}); // [num_windows*batch, window_size*window_size, embed_dim]
//     return {x, {batch, h / window_size, w / window_size}};
// }
//
// torch::Tensor window_reverse(torch::Tensor windows, int window_size, const std::vector<int64_t>& shape) {
//     // windows: [num_windows*batch, window_size*window_size, embed_dim]
//     int batch = shape[0], h_win = shape[1], w_win = shape[2];
//     windows = windows.view({batch, h_win, w_win, window_size, window_size, -1}) // [batch, h_win, w_win, window_size, window_size, embed_dim]
//                      .permute({0, 1, 3, 2, 4, 5}) // [batch, h_win, window_size, w_win, window_size, embed_dim]
//                      .contiguous()
//                      .view({batch, h_win * window_size, w_win * window_size, -1}); // [batch, h, w, embed_dim]
//     return windows.view({batch, -1, windows.size(-1)}); // [batch, num_patches, embed_dim]
// }
//
// // Window-based Multi-Head Self-Attention with Scaled Cosine Attention
// struct WMSAImpl : torch::nn::Module {
//     WMSAImpl(int embed_dim, int num_heads, int window_size, bool shift = false) : window_size_(window_size), shift_(shift) {
//         num_heads_ = num_heads;
//         head_dim_ = embed_dim / num_heads;
//         qkv = register_module("qkv", torch::nn::Linear(embed_dim, embed_dim * 3));
//         proj = register_module("proj", torch::nn::Linear(embed_dim, embed_dim));
//         log_attn_ratio = register_parameter("log_attn_ratio", torch::zeros({num_heads_}));
//         if (shift) {
//             // Simplified continuous positional bias (log-spaced)
//             pos_bias = register_parameter("pos_bias", torch::randn({2 * window_size - 1, 2 * window_size - 1}));
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, num_patches, embed_dim]
//         auto sizes = x.sizes();
//         int batch = sizes[0], num_patches = sizes[1];
//         int h = static_cast<int>(std::sqrt(num_patches)), w = h;
//
//         // Shift window if needed
//         torch::Tensor x_shifted = x;
//         if (shift_) {
//             x_shifted = torch::roll(x.view({batch, h, w, -1}), {-window_size_ / 2, -window_size_ / 2}, {1, 2})
//                            .view({batch, num_patches, -1});
//         }
//
//         // Window partition
//         auto [windows, window_shape] = window_partition(x_shifted, window_size_); // [num_windows*batch, window_size*window_size, embed_dim]
//
//         // QKV projection
//         auto qkv_out = qkv->forward(windows); // [num_windows*batch, window_size*window_size, embed_dim*3]
//         qkv_out = qkv_out.view({-1, window_size_ * window_size_, 3, num_heads_, head_dim_})
//                          .permute({2, 0, 3, 1, 4}); // [3, num_windows*batch, num_heads, window_size*window_size, head_dim]
//         auto q = qkv_out[0], k = qkv_out[1], v = qkv_out[2]; // [num_windows*batch, num_heads, window_size*window_size, head_dim]
//
//         // Scaled cosine attention
//         auto q_norm = q / (q.norm(2, -1, true) + 1e-6); // [num_windows*batch, num_heads, window_size*window_size, head_dim]
//         auto k_norm = k / (k.norm(2, -1, true) + 1e-6);
//         auto attn = torch::matmul(q_norm, k_norm.transpose(-2, -1)); // [num_windows*batch, num_heads, window_size*window_size, window_size*window_size]
//         attn = attn * log_attn_ratio.exp().view({1, num_heads_, 1, 1}); // Learnable scaling
//
//         // Add positional bias if shifted
//         if (shift_) {
//             // Simplified bias application (use center of window_size*window_size grid)
//             auto bias = pos_bias.slice(0, window_size_ - 1, window_size_)
//                                .slice(1, window_size_ - 1, window_size_)
//                                .view({1, 1, 1, 1});
//             attn = attn + bias;
//         }
//
//         attn = torch::softmax(attn, -1);
//         auto out = torch::matmul(attn, v); // [num_windows*batch, num_heads, window_size*window_size, head_dim]
//         out = out.permute({0, 2, 1, 3}).contiguous()
//                  .view({-1, window_size_ * window_size_, num_heads_ * head_dim_}); // [num_windows*batch, window_size*window_size, embed_dim]
//         out = proj->forward(out); // [num_windows*batch, window_size*window_size, embed_dim]
//
//         // Reverse window partition
//         out = window_reverse(out, window_size_, window_shape); // [batch, num_patches, embed_dim]
//
//         // Reverse shift if needed
//         if (shift_) {
//             out = torch::roll(out.view({batch, h, w, -1}), {window_size_ / 2, window_size_ / 2}, {1, 2})
//                      .view({batch, num_patches, -1});
//         }
//
//         return out;
//     }
//
//     int num_heads_, head_dim_, window_size_;
//     bool shift_;
//     torch::nn::Linear qkv{nullptr}, proj{nullptr};
//     torch::Tensor log_attn_ratio, pos_bias;
// };
// TORCH_MODULE(WMSA);
//
// // Swin Transformer Block
// struct SwinBlockImpl : torch::nn::Module {
//     SwinBlockImpl(int embed_dim, int num_heads, int window_size, bool shift = false, int mlp_ratio = 4, float dropout = 0.1) {
//         attn = register_module("attn", WMSA(embed_dim, num_heads, window_size, shift));
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
//         x = attn->forward(x);
//         x = norm1->forward(x + residual); // Post-layer norm
//         residual = x;
//         x = mlp->forward(x);
//         x = norm2->forward(x + residual); // Post-layer norm
//         return x;
//     }
//
//     WMSA attn{nullptr};
//     torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
//     torch::nn::Sequential mlp{nullptr};
// };
// TORCH_MODULE(SwinBlock);
//
// // Swin Transformer v2 Model
// struct SwinTransformerV2Impl : torch::nn::Module {
//     SwinTransformerV2Impl(int img_size, int patch_size, int in_channels, int num_classes, int window_size = 7) {
//         patch_embed = register_module("patch_embed", PatchEmbed(img_size, patch_size, in_channels, 96));
//
//         // Stage 1: embed_dim=96, num_patches=16
//         stage1->push_back(SwinBlock(96, 3, window_size, false));
//         stage1->push_back(SwinBlock(96, 3, window_size, true));
//         register_module("stage1_block_0", stage1[0]);
//         register_module("stage1_block_1", stage1[1]);
//
//         // Stage 2: embed_dim=192, num_patches=4
//         patch_merge1 = register_module("patch_merge1", torch::nn::Linear(96 * 4, 192));
//         stage2->push_back(SwinBlock(192, 6, window_size, false));
//         stage2->push_back(SwinBlock(192, 6, window_size, true));
//         register_module("stage2_block_0", stage2[0]);
//         register_module("stage2_block_1", stage2[1]);
//
//         // Head
//         norm = register_module("norm", torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({192}).eps(1e-6)));
//         head = register_module("head", torch::nn::Linear(192, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, in_channels, img_size, img_size]
//         x = patch_embed->forward(x); // [batch, 16, 96]
//
//         // Stage 1
//         for (auto& block : *stage1) {
//             x = block->forward(x); // [batch, 16, 96]
//         }
//
//         // Patch merging
//         int batch = x.size(0), seq_len = x.size(1);
//         int h = static_cast<int>(std::sqrt(seq_len)); // 4
//         x = x.view({batch, h, h, -1}) // [batch, 4, 4, 96]
//              .transpose(2, 3) // [batch, 4, 96, 4]
//              .contiguous()
//              .view({batch, h, -1}); // [batch, 4, 96*4]
//         x = x.view({batch, h / 2, 2, -1}) // [batch, 2, 2, 96*4]
//              .transpose(2, 3) // [batch, 2, 96*4, 2]
//              .contiguous()
//              .view({batch, -1, 96 * 4}); // [batch, 4, 96*4]
//         x = patch_merge1->forward(x); // [batch, 4, 192]
//
//         // Stage 2
//         for (auto& block : *stage2) {
//             x = block->forward(x); // [batch, 4, 192]
//         }
//
//         // Pool and classify
//         x = norm->forward(x);
//         x = x.mean(1); // [batch, 192]
//         x = head->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     PatchEmbed patch_embed{nullptr};
//     torch::nn::ModuleList stage1{nullptr}, stage2{nullptr};
//     torch::nn::Linear patch_merge1{nullptr}, head{nullptr};
//     torch::nn::LayerNorm norm{nullptr};
// };
// TORCH_MODULE(SwinTransformerV2);
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
//         const int patch_size = 7;
//         const int in_channels = 1;
//         const int num_classes = 10; // e.g., MNIST digits
//         const int window_size = 4; // Adjusted for small input
//         const int batch_size = 16;
//         const float lr = 0.001;
//         const int num_epochs = 20;
//
//         // Initialize model
//         SwinTransformerV2 model(img_size, patch_size, in_channels, num_classes, window_size);
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
//                 torch::save(model, "swin_v2_epoch_" + std::to_string(epoch + 1) + ".pt");
//             }
//         }
//
//         // Save final model
//         torch::save(model, "swin_v2.pt");
//         std::cout << "Model saved as swin_v2.pt" << std::endl;
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
    SwinTransformerV1::SwinTransformerV1(int num_classes, int in_channels)
    {
    }

    SwinTransformerV1::SwinTransformerV1(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void SwinTransformerV1::reset()
    {
    }

    auto SwinTransformerV1::forward(std::initializer_list<std::any> tensors) -> std::any
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

    SwinTransformerV2::SwinTransformerV2(int num_classes, int in_channels)
    {
    }

    SwinTransformerV2::SwinTransformerV2(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void SwinTransformerV2::reset()
    {
    }

    auto SwinTransformerV2::forward(std::initializer_list<std::any> tensors) -> std::any
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
