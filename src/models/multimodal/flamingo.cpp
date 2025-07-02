#include "include/models/multimodal/flamingo.h"


using namespace std;


// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <iostream>
// #include <vector>
// #include <string>
// #include <fstream>
// #include <map>
//
// // Simple Vocabulary Class
// class Vocabulary {
// public:
//     Vocabulary(const std::string& vocab_file) {
//         std::ifstream file(vocab_file);
//         std::string token;
//         int idx = 0;
//         while (std::getline(file, token)) {
//             token_to_idx_[token] = idx;
//             idx_to_token_[idx] = token;
//             idx++;
//         }
//         vocab_size_ = idx;
//     }
//
//     int token_to_idx(const std::string& token) const {
//         auto it = token_to_idx_.find(token);
//         return it != token_to_idx_.end() ? it->second : token_to_idx_.at("<unk>");
//     }
//
//     std::string idx_to_token(int idx) const {
//         auto it = idx_to_token_.find(idx);
//         return it != idx_to_token_.end() ? it->second : "<unk>";
//     }
//
//     int vocab_size() const { return vocab_size_; }
//
// private:
//     std::map<std::string, int> token_to_idx_;
//     std::map<int, std::string> idx_to_token_;
//     int vocab_size_;
// };
//
// // Transformer Encoder Layer
// struct TransformerEncoderLayerImpl : torch::nn::Module {
//     TransformerEncoderLayerImpl(int d_model, int nhead, int dim_feedforward, float dropout) {
//         self_attn = register_module("self_attn", torch::nn::MultiheadAttention(
//             torch::nn::MultiheadAttentionOptions(d_model, nhead).dropout(dropout)));
//         linear1 = register_module("linear1", torch::nn::Linear(d_model, dim_feedforward));
//         dropout1 = register_module("dropout1", torch::nn::Dropout(dropout));
//         linear2 = register_module("linear2", torch::nn::Linear(dim_feedforward, d_model));
//         norm1 = register_module("norm1", torch::nn::LayerNorm(d_model));
//         norm2 = register_module("norm2", torch::nn::LayerNorm(d_model));
//         dropout2 = register_module("dropout2", torch::nn::Dropout(dropout));
//     }
//
//     torch::Tensor forward(torch::Tensor src, torch::Tensor src_mask = {}) {
//         auto attn_out = self_attn->forward(src, src, src, {}, src_mask);
//         src = norm1->forward(src + dropout1->forward(std::get<0>(attn_out)));
//         auto ff_out = linear2->forward(torch::relu(linear1->forward(src)));
//         src = norm2->forward(src + dropout2->forward(ff_out));
//         return src;
//     }
//
//     torch::nn::MultiheadAttention self_attn{nullptr};
//     torch::nn::Linear linear1{nullptr}, linear2{nullptr};
//     torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr};
//     torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
// };
// TORCH_MODULE(TransformerEncoderLayer);
//
// // Transformer Encoder
// struct TransformerEncoderImpl : torch::nn::Module {
//     TransformerEncoderImpl(int d_model, int nhead, int num_layers, int dim_feedforward, float dropout) {
//         for (int i = 0; i < num_layers; ++i) {
//             layers->push_back(TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout));
//             register_module("layer_" + std::to_string(i), layers->back());
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor src, torch::Tensor src_mask = {}) {
//         torch::Tensor output = src;
//         for (auto& layer : *layers) {
//             output = layer->forward(output, src_mask);
//         }
//         return output;
//     }
//
//     torch::nn::ModuleList layers{torch::nn::ModuleList()};
// };
// TORCH_MODULE(TransformerEncoder);
//
// // Perceiver Resampler
// struct PerceiverResamplerImpl : torch::nn::Module {
//     PerceiverResamplerImpl(int d_model, int num_latents, int depth, int nhead, int dim_feedforward, float dropout) {
//         latent_tokens = register_parameter("latent_tokens", torch::randn({num_latents, d_model}));
//         for (int i = 0; i < depth; ++i) {
//             layers->push_back(TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout));
//             register_module("layer_" + std::to_string(i), layers->back());
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor media) {
//         // media: [batch, seq_len, d_model]
//         auto batch_size = media.size(0);
//         auto latents = latent_tokens.unsqueeze(0).repeat({batch_size, 1, 1}); // [batch, num_latents, d_model]
//         auto input = torch::cat({latents, media}, 1); // [batch, num_latents + seq_len, d_model]
//         return layers->forward(input).slice(1, 0, latent_tokens.size(0)); // [batch, num_latents, d_model]
//     }
//
//     torch::Tensor latent_tokens;
//     torch::nn::ModuleList layers{torch::nn::ModuleList()};
// };
// TORCH_MODULE(PerceiverResampler);
//
// // Gated Cross-Attention Block
// struct GatedCrossAttentionBlockImpl : torch::nn::Module {
//     GatedCrossAttentionBlockImpl(int d_model, int nhead, int dim_feedforward, float dropout) {
//         cross_attn = register_module("cross_attn", torch::nn::MultiheadAttention(
//             torch::nn::MultiheadAttentionOptions(d_model, nhead).dropout(dropout)));
//         gate = register_parameter("gate", torch::zeros({1}));
//         linear1 = register_module("linear1", torch::nn::Linear(d_model, dim_feedforward));
//         dropout1 = register_module("dropout1", torch::nn::Dropout(dropout));
//         linear2 = register_module("linear2", torch::nn::Linear(dim_feedforward, d_model));
//         norm1 = register_module("norm1", torch::nn::LayerNorm(d_model));
//         norm2 = register_module("norm2", torch::nn::LayerNorm(d_model));
//         dropout2 = register_module("dropout2", torch::nn::Dropout(dropout));
//     }
//
//     torch::Tensor forward(torch::Tensor text, torch::Tensor media, torch::Tensor media_locations = {}) {
//         if (!media_locations.defined()) {
//             media_locations = torch::zeros({text.size(0), text.size(1)}, torch::kBool).to(text.device());
//         }
//         auto attn_out = cross_attn->forward(text, media, media);
//         auto gated_attn = torch::tanh(gate) * std::get<0>(attn_out);
//         text = norm1->forward(text + dropout1->forward(gated_attn));
//         auto ff_out = linear2->forward(torch::relu(linear1->forward(text)));
//         text = norm2->forward(text + dropout2->forward(ff_out));
//         return text;
//     }
//
//     torch::nn::MultiheadAttention cross_attn{nullptr};
//     torch::Tensor gate;
//     torch::nn::Linear linear1{nullptr}, linear2{nullptr};
//     torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr};
//     torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
// };
// TORCH_MODULE(GatedCrossAttentionBlock);
//
// // Transformer Decoder Layer with Gated Cross-Attention
// struct TransformerDecoderLayerImpl : torch::nn::Module {
//     TransformerDecoderLayerImpl(int d_model, int nhead, int dim_feedforward, float dropout) {
//         self_attn = register_module("self_attn", torch::nn::MultiheadAttention(
//             torch::nn::MultiheadAttentionOptions(d_model, nhead).dropout(dropout)));
//         cross_attn_block = register_module("cross_attn_block", GatedCrossAttentionBlock(d_model, nhead, dim_feedforward, dropout));
//         norm1 = register_module("norm1", torch::nn::LayerNorm(d_model));
//         norm2 = register_module("norm2", torch::nn::LayerNorm(d_model));
//     }
//
//     torch::Tensor forward(torch::Tensor tgt, torch::Tensor memory, torch::Tensor tgt_mask = {}, torch::Tensor media_locations = {}) {
//         auto self_attn_out = self_attn->forward(tgt, tgt, tgt, {}, tgt_mask);
//         tgt = norm1->forward(tgt + self_attn_out.first);
//         tgt = cross_attn_block->forward(tgt, memory, media_locations);
//         return tgt;
//     }
//
//     torch::nn::MultiheadAttention self_attn{nullptr};
//     GatedCrossAttentionBlock cross_attn_block{nullptr};
//     torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
// };
// TORCH_MODULE(TransformerDecoderLayer);
//
// // Transformer Decoder
// struct TransformerDecoderImpl : torch::nn::Module {
//     TransformerDecoderImpl(int d_model, int nhead, int num_layers, int dim_feedforward, float dropout) {
//         for (int i = 0; i < num_layers; ++i) {
//             layers->push_back(TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout));
//             register_module("layer_" + std::to_string(i), layers->back());
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor tgt, torch::Tensor memory, torch::Tensor tgt_mask = {}, torch::Tensor media_locations = {}) {
//         torch::Tensor output = tgt;
//         for (auto& layer : *layers) {
//             output = layer->forward(output, memory, tgt_mask, media_locations);
//         }
//         return output;
//     }
//
//     torch::nn::ModuleList layers{torch::nn::ModuleList()};
// };
// TORCH_MODULE(TransformerDecoder);
//
// // Simplified Flamingo-like Model
// struct FlamingoImpl : torch::nn::Module {
//     FlamingoImpl(int vocab_size, int d_model = 256, int nhead = 4, int num_layers = 2, int dim_feedforward = 512, int num_latents = 64, float dropout = 0.1) {
//         // Vision Transformer (simplified)
//         vit_conv = register_module("vit_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(1, d_model, 16).stride(16))); // Patch embedding for 224x224
//         vit_norm = register_module("vit_norm", torch::nn::LayerNorm(d_model));
//         vit_positional_encoding = register_parameter("vit_positional_encoding",
//             torch::randn({1, 197, d_model})); // 196 patches + CLS token
//         vit_encoder = register_module("vit_encoder", TransformerEncoder(d_model, nhead, num_layers, dim_feedforward, dropout));
//
//         // Perceiver Resampler
//         perceiver = register_module("perceiver", PerceiverResampler(d_model, num_latents, 2, nhead, dim_feedforward, dropout));
//
//         // Text Decoder
//         token_embedding = register_module("token_embedding", torch::nn::Embedding(vocab_size, d_model));
//         text_positional_encoding = register_parameter("text_positional_encoding",
//             torch::randn({1, 50, d_model})); // Max text length 50
//         decoder = register_module("decoder", TransformerDecoder(d_model, nhead, num_layers, dim_feedforward, dropout));
//         output_layer = register_module("output_layer", torch::nn::Linear(d_model, vocab_size));
//     }
//
//     torch::Tensor forward(torch::Tensor image, torch::Tensor captions, torch::Tensor tgt_mask = {}, torch::Tensor media_locations = {}) {
//         // Image encoding
//         auto img_features = vit_conv->forward(image); // [batch, d_model, 14, 14]
//         img_features = img_features.view({image.size(0), d_model, -1}).permute({0, 2, 1}); // [batch, 196, d_model]
//         auto cls_token = torch::zeros({image.size(0), 1, d_model}, image.options()).to(image.device());
//         img_features = torch::cat({cls_token, img_features}, 1); // [batch, 197, d_model]
//         img_features = img_features + vit_positional_encoding;
//         img_features = vit_norm->forward(img_features);
//         img_features = vit_encoder->forward(img_features); // [batch, 197, d_model]
//
//         // Perceiver Resampler
//         auto perceived = perceiver->forward(img_features); // [batch, num_latents, d_model]
//
//         // Caption embedding
//         auto tgt = token_embedding->forward(captions); // [batch, seq_len, d_model]
//         tgt = tgt + text_positional_encoding.slice(1, 0, captions.size(1));
//
//         // Decoder
//         auto output = decoder->forward(tgt, perceived, tgt_mask, media_locations); // [batch, seq_len, d_model]
//
//         // Output logits
//         return output_layer->forward(output); // [batch, seq_len, vocab_size]
//     }
//
//     std::string generate_caption(torch::Tensor image, const Vocabulary& vocab, int max_length = 20) {
//         model()->eval();
//         torch::NoGradGuard no_grad;
//
//         // Image encoding
//         auto img_features = vit_conv->forward(image); // [batch, d_model, 14, 14]
//         img_features = img_features.view({image.size(0), d_model, -1}).permute({0, 2, 1}); // [batch, 196, d_model]
//         auto cls_token = torch::zeros({image.size(0), 1, d_model}, image.options()).to(image.device());
//         img_features = torch::cat({cls_token, img_features}, 1); // [batch, 197, d_model]
//         img_features = img_features + vit_positional_encoding;
//         img_features = vit_norm->forward(img_features);
//         img_features = vit_encoder->forward(img_features); // [batch, 197, d_model]
//
//         // Perceiver Resampler
//         auto perceived = perceiver->forward(img_features); // [batch, num_latents, d_model]
//
//         // Start with <sos> token
//         std::vector<int64_t> caption = {vocab.token_to_idx("<sos>")};
//         torch::Tensor tgt = torch::tensor(caption, torch::kInt64).unsqueeze(0).to(image.device()); // [1, 1]
//
//         for (int i = 0; i < max_length; ++i) {
//             auto tgt_emb = token_embedding->forward(tgt); // [1, seq_len, d_model]
//             tgt_emb = tgt_emb + text_positional_encoding.slice(1, 0, tgt.size(1));
//             auto output = decoder->forward(tgt_emb, perceived); // [1, seq_len, d_model]
//             auto logits = output_layer->forward(output.slice(1, -1)); // [1, vocab_size]
//             auto next_token = logits.argmax(-1).item<int64_t>();
//
//             caption.push_back(next_token);
//             if (vocab.idx_to_token(next_token) == "<eos>") break;
//
//             tgt = torch::tensor(caption, torch::kInt64).unsqueeze(0).to(image.device());
//         }
//
//         std::string result;
//         for (int idx : caption) {
//             result += vocab.idx_to_token(idx) + " ";
//         }
//         return result;
//     }
//
//     torch::nn::Conv2d vit_conv{nullptr};
//     torch::nn::LayerNorm vit_norm{nullptr};
//     torch::Tensor vit_positional_encoding, text_positional_encoding;
//     TransformerEncoder vit_encoder{nullptr};
//     PerceiverResampler perceiver{nullptr};
//     torch::nn::Embedding token_embedding{nullptr};
//     TransformerDecoder decoder{nullptr};
//     torch::nn::Linear output_layer{nullptr};
// };
// TORCH_MODULE(Flamingo);
//
// // Load Image
// torch::Tensor load_image(const std::string& path) {
//     cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
//     if (image.empty()) {
//         throw std::runtime_error("Failed to load image: " + path);
//     }
//     cv::resize(image, image, cv::Size(224, 224));
//     image.convertTo(image, CV_32F, 2.0 / 255.0, -1.0); // Normalize to [-1, 1]
//     torch::Tensor img_tensor = torch::from_blob(image.data, {1, 1, 224, 224}, torch::kFloat32);
//     return img_tensor;
// }
//
// int main() {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Load vocabulary
//         Vocabulary vocab("./vocab.txt");
//
//         // Initialize model
//         Flamingo model(vocab.vocab_size());
//         model->to(device);
//
//         // Load pre-trained weights (assumed to be saved as a .pt file)
//         try {
//             torch::load(model, "flamingo.pt");
//             std::cout << "Loaded pre-trained model weights from flamingo.pt" << std::endl;
//         } catch (const std::exception& e) {
//             std::cerr << "Failed to load model weights: " << e.what() << std::endl;
//             return -1;
//         }
//
//         // Load and preprocess image
//         torch::Tensor image = load_image("./data/test_image.jpg").to(device);
//
//         // Generate caption
//         std::string caption = model->generate_caption(image, vocab);
//         std::cout << "Generated Caption: " << caption << std::endl;
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
    Flamingo::Flamingo(int num_classes, int in_channels)
    {
    }

    Flamingo::Flamingo(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void Flamingo::reset()
    {
    }

    auto Flamingo::forward(std::initializer_list<std::any> tensors) -> std::any
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
