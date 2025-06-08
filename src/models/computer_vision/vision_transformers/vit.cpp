#include "include/models/computer_vision/vision_transformers/vit.h"


using namespace std;

//ViT GROK


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
// // Multi-Head Self-Attention (MSA)
// struct MSAImpl : torch::nn::Module {
//     MSAImpl(int embed_dim, int num_heads, float dropout = 0.1) {
//         num_heads_ = num_heads;
//         head_dim_ = embed_dim / num_heads;
//         scale_ = 1.0 / std::sqrt(head_dim_);
//         qkv = register_module("qkv", torch::nn::Linear(embed_dim, embed_dim * 3));
//         proj = register_module("proj", torch::nn::Linear(embed_dim, embed_dim));
//         dropout_layer = register_module("dropout", torch::nn::Dropout(dropout));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, seq_len, embed_dim]
//         auto sizes = x.sizes();
//         int batch = sizes[0], seq_len = sizes[1];
//
//         // QKV projection
//         auto qkv_out = qkv->forward(x); // [batch, seq_len, embed_dim*3]
//         qkv_out = qkv_out.view({batch, seq_len, 3, num_heads_, head_dim_})
//                          .permute({2, 0, 3, 1, 4}); // [3, batch, num_heads, seq_len, head_dim]
//         auto q = qkv_out[0], k = qkv_out[1], v = qkv_out[2]; // [batch, num_heads, seq_len, head_dim]
//
//         // Attention
//         auto attn = torch::matmul(q, k.transpose(-2, -1)) * scale_; // [batch, num_heads, seq_len, seq_len]
//         attn = torch::softmax(attn, -1);
//         attn = dropout_layer->forward(attn);
//         auto out = torch::matmul(attn, v); // [batch, num_heads, seq_len, head_dim]
//         out = out.permute({0, 2, 1, 3}).contiguous()
//                  .view({batch, seq_len, num_heads_ * head_dim_}); // [batch, seq_len, embed_dim]
//         out = proj->forward(out);
//         out = dropout_layer->forward(out);
//         return out;
//     }
//
//     int num_heads_, head_dim_;
//     float scale_;
//     torch::nn::Linear qkv{nullptr}, proj{nullptr};
//     torch::nn::Dropout dropout_layer{nullptr};
// };
// TORCH_MODULE(MSA);
//
// // Feed-Forward Network (FFN)
// struct FFNImpl : torch::nn::Module {
//     FFNImpl(int embed_dim, int mlp_ratio = 4, float dropout = 0.1) {
//         fc1 = register_module("fc1", torch::nn::Linear(embed_dim, embed_dim * mlp_ratio));
//         fc2 = register_module("fc2", torch::nn::Linear(embed_dim * mlp_ratio, embed_dim));
//         dropout1 = register_module("dropout1", torch::nn::Dropout(dropout));
//         dropout2 = register_module("dropout2", torch::nn::Dropout(dropout));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, seq_len, embed_dim]
//         x = fc1->forward(x); // [batch, seq_len, embed_dim*mlp_ratio]
//         x = torch::gelu(x);
//         x = dropout1->forward(x);
//         x = fc2->forward(x); // [batch, seq_len, embed_dim]
//         x = dropout2->forward(x);
//         return x;
//     }
//
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr};
//     torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr};
// };
// TORCH_MODULE(FFN);
//
// // Transformer Encoder Block
// struct EncoderBlockImpl : torch::nn::Module {
//     EncoderBlockImpl(int embed_dim, int num_heads, int mlp_ratio = 4, float dropout = 0.1) {
//         norm1 = register_module("norm1", torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({embed_dim}).eps(1e-6)));
//         attn = register_module("attn", MSA(embed_dim, num_heads, dropout));
//         norm2 = register_module("norm2", torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({embed_dim}).eps(1e-6)));
//         ffn = register_module("ffn", FFN(embed_dim, mlp_ratio, dropout));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, seq_len, embed_dim]
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
//     torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
//     MSA attn{nullptr};
//     FFN ffn{nullptr};
// };
// TORCH_MODULE(EncoderBlock);
//
// // Vision Transformer (ViT) Model
// struct ViTImpl : torch::nn::Module {
//     ViTImpl(int img_size, int patch_size, int in_channels, int num_classes, int embed_dim, int num_heads, int depth) {
//         num_patches_ = (img_size / patch_size) * (img_size / patch_size);
//         patch_embed = register_module("patch_embed", PatchEmbed(img_size, patch_size, in_channels, embed_dim));
//         cls_token = register_parameter("cls_token", torch::randn({1, 1, embed_dim}));
//         pos_embed = register_parameter("pos_embed", torch::randn({1, num_patches_ + 1, embed_dim}));
//
//         // Encoder blocks
//         for (int i = 0; i < depth; ++i) {
//             encoder->push_back(EncoderBlock(embed_dim, num_heads));
//             register_module("encoder_block_" + std::to_string(i), encoder[i]);
//         }
//
//         // Head
//         norm = register_module("norm", torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({embed_dim}).eps(1e-6)));
//         head = register_module("head", torch::nn::Linear(embed_dim, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, in_channels, img_size, img_size]
//         int batch = x.size(0);
//
//         // Patch embedding
//         x = patch_embed->forward(x); // [batch, num_patches, embed_dim]
//
//         // Add cls token
//         auto cls_tokens = cls_token.expand({batch, -1, -1}); // [batch, 1, embed_dim]
//         x = torch::cat({cls_tokens, x}, 1); // [batch, num_patches+1, embed_dim]
//
//         // Add positional embedding
//         x = x + pos_embed; // [batch, num_patches+1, embed_dim]
//
//         // Encoder
//         for (auto& block : *encoder) {
//             x = block->forward(x); // [batch, num_patches+1, embed_dim]
//         }
//
//         // Classification
//         x = norm->forward(x);
//         x = x.slice(1, 0, 1).squeeze(1); // [batch, embed_dim] (cls token)
//         x = head->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     int num_patches_;
//     PatchEmbed patch_embed{nullptr};
//     torch::nn::ModuleList encoder{nullptr};
//     torch::nn::LayerNorm norm{nullptr};
//     torch::nn::Linear head{nullptr};
//     torch::Tensor cls_token, pos_embed;
// };
// TORCH_MODULE(ViT);
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
//         const int embed_dim = 96;
//         const int num_heads = 3;
//         const int depth = 6;
//         const int batch_size = 16;
//         const float lr = 0.001;
//         const int num_epochs = 20;
//
//         // Initialize model
//         ViT model(img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, depth);
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
//                 torch::save(model, "vit_epoch_" + std::to_string(epoch + 1) + ".pt");
//             }
//         }
//
//         // Save final model
//         torch::save(model, "vit.pt");
//         std::cout << "Model saved as vit.pt" << std::endl;
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
    ViT::ViT(int num_classes, int in_channels)
    {
    }

    ViT::ViT(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void ViT::reset()
    {
    }

    auto ViT::forward(std::initializer_list<std::any> tensors) -> std::any
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
