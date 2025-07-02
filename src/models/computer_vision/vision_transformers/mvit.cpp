#include "include/models/computer_vision/vision_transformers/mvit.h"


using namespace std;

//MViT GROK



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
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, in_channels, img_size, img_size]
//         x = proj->forward(x); // [batch, embed_dim, img_size/patch_size, img_size/patch_size]
//         x = x.flatten(2).transpose(1, 2); // [batch, num_patches, embed_dim]
//         return x;
//     }
//
//     int num_patches_;
//     torch::nn::Conv2d proj{nullptr};
// };
// TORCH_MODULE(PatchEmbed);
//
// // Multi-Head Pooling Attention (MHPA)
// struct MHPAImpl : torch::nn::Module {
//     MHPAImpl(int embed_dim, int num_heads, int pool_stride = 1) {
//         num_heads_ = num_heads;
//         head_dim_ = embed_dim / num_heads;
//         scale_ = 1.0 / std::sqrt(head_dim_);
//         qkv = register_module("qkv", torch::nn::Linear(embed_dim, embed_dim * 3));
//         proj = register_module("proj", torch::nn::Linear(embed_dim, embed_dim));
//         if (pool_stride > 1) {
//             pool = register_module("pool", torch::nn::AvgPool2d(
//                 torch::nn::AvgPool2dOptions(pool_stride).stride(pool_stride)));
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, seq_len, embed_dim]
//         auto sizes = x.sizes();
//         int batch = sizes[0], seq_len = sizes[1];
//         auto qkv_out = qkv->forward(x); // [batch, seq_len, embed_dim * 3]
//         qkv_out = qkv_out.view({batch, seq_len, 3, num_heads_, head_dim_})
//                           .permute({2, 0, 3, 1, 4}); // [3, batch, num_heads, seq_len, head_dim]
//         auto q = qkv_out[0], k = qkv_out[1], v = qkv_out[2]; // [batch, num_heads, seq_len, head_dim]
//
//         // Pooling for query (simplified for image case)
//         if (pool.defined()) {
//             auto q_reshaped = q.permute({0, 1, 3, 2}) // [batch, num_heads, head_dim, seq_len]
//                               .view({batch * num_heads_, head_dim_, static_cast<int>(std::sqrt(seq_len)), -1}); // [batch*num_heads, head_dim, sqrt(seq_len), sqrt(seq_len)]
//             q_reshaped = pool->forward(q_reshaped); // [batch*num_heads, head_dim, sqrt(seq_len)/stride, sqrt(seq_len)/stride]
//             auto new_seq_len = q_reshaped.size(2) * q_reshaped.size(3);
//             q = q_reshaped.view({batch, num_heads_, head_dim_, new_seq_len})
//                           .permute({0, 1, 3, 2}); // [batch, num_heads, new_seq_len, head_dim]
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
//     int num_heads_, head_dim_;
//     float scale_;
//     torch::nn::Linear qkv{nullptr}, proj{nullptr};
//     torch::nn::AvgPool2d pool{nullptr};
// };
// TORCH_MODULE(MHPA);
//
// // Transformer Block with MHPA
// struct TransformerBlockImpl : torch::nn::Module {
//     TransformerBlockImpl(int embed_dim, int num_heads, int mlp_ratio = 4, int pool_stride = 1, float dropout = 0.1) {
//         attn = register_module("attn", MHPA(embed_dim, num_heads, pool_stride));
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
//     MHPA attn{nullptr};
//     torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
//     torch::nn::Sequential mlp{nullptr};
// };
// TORCH_MODULE(TransformerBlock);
//
// // MViT Model
// struct MViTImpl : torch::nn::Module {
//     MViTImpl(int img_size, int patch_size, int in_channels, int num_classes) {
//         patch_embed = register_module("patch_embed", PatchEmbed(img_size, patch_size, in_channels, 96));
//         pos_embed = register_parameter("pos_embed", torch::randn({1, patch_embed->num_patches_, 96}));
//
//         // Stage 1: 4 blocks, embed_dim=96, seq_len=16
//         for (int i = 0; i < 4; ++i) {
//             stage1->push_back(TransformerBlock(96, 4));
//             register_module("stage1_block_" + std::to_string(i), stage1[i]);
//         }
//
//         // Stage 2: 4 blocks, embed_dim=192, seq_len=4 (pool stride=2)
//         stage2_transition = register_module("stage2_transition", torch::nn::Linear(96, 192));
//         for (int i = 0; i < 4; ++i) {
//             stage2->push_back(TransformerBlock(192, 8, 4, i == 0 ? 2 : 1));
//             register_module("stage2_block_" + std::to_string(i), stage2[i]);
//         }
//
//         // Head
//         norm = register_module("norm", torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({192}).eps(1e-6)));
//         head = register_module("head", torch::nn::Linear(192, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, in_channels, img_size, img_size]
//         x = patch_embed->forward(x); // [batch, num_patches, 96]
//         x = x + pos_embed;
//
//         // Stage 1
//         for (auto& block : *stage1) {
//             x = block->forward(x); // [batch, 16, 96]
//         }
//
//         // Stage 2
//         x = stage2_transition->forward(x); // [batch, 16, 192]
//         for (auto& block : *stage2) {
//             x = block->forward(x); // [batch, 4, 192] after first block
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
//     torch::Tensor pos_embed;
//     torch::nn::ModuleList stage1{nullptr}, stage2{nullptr};
//     torch::nn::Linear stage2_transition{nullptr}, head{nullptr};
//     torch::nn::LayerNorm norm{nullptr};
// };
// TORCH_MODULE(MViT);
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
//         const int batch_size = 16;
//         const float lr = 0.001;
//         const int num_epochs = 20;
//
//         // Initialize model
//         MViT model(img_size, patch_size, in_channels, num_classes);
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
//                 torch::save(model, "mvit_epoch_" + std::to_string(epoch + 1) + ".pt");
//             }
//         }
//
//         // Save final model
//         torch::save(model, "mvit.pt");
//         std::cout << "Model saved as mvit.pt" << std::endl;
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
    MViT::MViT(int num_classes, int in_channels)
    {
    }

    MViT::MViT(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void MViT::reset()
    {
    }

    auto MViT::forward(std::initializer_list<std::any> tensors) -> std::any
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
