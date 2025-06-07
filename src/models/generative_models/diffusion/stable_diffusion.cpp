#include "include/models/generative_models/diffusion/stable_diffusion.h"


using namespace std;


//GMINI

// #pragma once
// #include <torch/torch.h>
// #include <vector>
//
// // Simplified Residual Block
// struct ResBlockImpl : torch::nn::Module {
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
//     torch::nn::GroupNorm gn1{nullptr}, gn2{nullptr};
//     torch::nn::SiLU silu{nullptr};
//     torch::nn::Conv2d nin_shortcut{nullptr};
//
//     ResBlockImpl(int64_t in_channels, int64_t out_channels, int64_t groups1 = 32, int64_t groups2 = 32) {
//         if (in_channels > 0) { // Basic check
//             gn1 = torch::nn::GroupNorm(torch::nn::GroupNormOptions(std::min(groups1, in_channels), in_channels));
//             register_module("gn1", gn1);
//         }
//         silu = torch::nn::SiLU();
//         conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1));
//         if (out_channels > 0) {
//             gn2 = torch::nn::GroupNorm(torch::nn::GroupNormOptions(std::min(groups2, out_channels), out_channels));
//             register_module("gn2", gn2);
//         }
//         conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1));
//
//         if (in_channels != out_channels) {
//             nin_shortcut = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1));
//             register_module("nin_shortcut", nin_shortcut);
//         }
//         register_module("silu", silu);
//         register_module("conv1", conv1);
//         register_module("conv2", conv2);
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         torch::Tensor h = x;
//         if (gn1) h = gn1(h);
//         h = silu(h);
//         h = conv1(h);
//         if (gn2) h = gn2(h);
//         h = silu(h);
//         h = conv2(h);
//         if (nin_shortcut) x = nin_shortcut(x);
//         return x + h;
//     }
// };
// TORCH_MODULE(ResBlock);
//
// // Simplified Downsample Block
// struct DownsampleImpl : torch::nn::Module {
//     torch::nn::Conv2d conv{nullptr};
//     DownsampleImpl(int64_t channels)
//         : conv(torch::nn::Conv2dOptions(channels, channels, 3).stride(2).padding(1)) {
//         register_module("conv", conv);
//     }
//     torch::Tensor forward(torch::Tensor x) { return conv(x); }
// };
// TORCH_MODULE(Downsample);
//
// // Simplified Upsample Block
// struct UpsampleImpl : torch::nn::Module {
//     torch::nn::Conv2d conv{nullptr};
//     UpsampleImpl(int64_t channels)
//         : conv(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)) {
//         register_module("conv", conv);
//     }
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::upsample_nearest2d(x, /*output_size=*/c10::nullopt, /*scale_factors=*/std::vector<double>({2.0, 2.0}));
//         return conv(x);
//     }
// };
// TORCH_MODULE(Upsample);
//
// // Placeholder for Attention (massively simplified)
// struct AttentionBlockImpl : torch::nn::Module {
//     int64_t channels_;
//     torch::nn::GroupNorm norm{nullptr};
//     torch::nn::Conv2d q_conv{nullptr}, k_conv{nullptr}, v_conv{nullptr}, proj_out{nullptr};
//
//     AttentionBlockImpl(int64_t channels, int64_t groups = 32) : channels_(channels) {
//         if (channels > 0) {
//             norm = torch::nn::GroupNorm(torch::nn::GroupNormOptions(std::min(groups, channels), channels));
//             q_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 1));
//             k_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 1));
//             v_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 1));
//             proj_out = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 1));
//             register_module("norm", norm);
//             register_module("q_conv", q_conv);
//             register_module("k_conv", k_conv);
//             register_module("v_conv", v_conv);
//             register_module("proj_out", proj_out);
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x, torch::Tensor context = torch::Tensor()) {
//         if (!norm) return x; // Not initialized
//         torch::Tensor h_ = norm(x);
//         torch::Tensor q = q_conv(h_);
//         torch::Tensor k = k_conv(h_); // Self-attention
//         torch::Tensor v = v_conv(h_); // Self-attention
//         // Actual attention (q @ k.T / sqrt(dim)) @ v is missing for brevity.
//         // This is NOT real attention.
//         h_ = q * k * v;
//         h_ = proj_out(h_);
//         return x + h_;
//     }
// };
// TORCH_MODULE(AttentionBlock);
//
// // Sinusoidal timestep embedding
// inline torch::Tensor get_timestep_embedding(torch::Tensor timesteps, int embedding_dim) {
//     TORCH_CHECK(timesteps.dim() == 1, "Timesteps should be a 1D tensor");
//     int half_dim = embedding_dim / 2;
//     torch::Tensor freqs = torch::exp(
//         -std::log(10000.0) * torch::arange(0, half_dim, timesteps.options().dtype(torch::kFloat32)) / (half_dim -1)
//     );
//     torch::Tensor args = timesteps.to(torch::kFloat32).unsqueeze(1) * freqs.unsqueeze(0);
//     torch::Tensor embedding = torch::cat({torch::cos(args), torch::sin(args)}, /*dim=*/-1);
//     if (embedding_dim % 2 == 1) {
//         embedding = torch::cat({embedding, torch::zeros_like(embedding.slice(/*dim=*/1, /*start=*/0, /*end=*/1))}, /*dim=*/-1);
//     }
//     return embedding;
// }
//


// #pragma once
// #include "common_modules.h"
//
// struct SimplifiedUNetImpl : torch::nn::Module {
//     torch::nn::Conv2d in_conv{nullptr};
//     torch::nn::Linear time_proj1{nullptr}, time_proj2{nullptr};
//
//     ResBlock down_res1{nullptr}, down_res2{nullptr};
//     AttentionBlock down_attn1{nullptr};
//     Downsample downsampler1{nullptr};
//
//     ResBlock mid_res1{nullptr}; AttentionBlock mid_attn{nullptr};
//
//     Upsample upsampler1{nullptr};
//     ResBlock up_res1{nullptr}, up_res2{nullptr};
//     AttentionBlock up_attn1{nullptr};
//
//     torch::nn::GroupNorm out_norm{nullptr};
//     torch::nn::SiLU out_silu{nullptr};
//     torch::nn::Conv2d out_conv{nullptr};
//
//     int64_t model_channels = 64; // Very small for example
//     int64_t time_embed_dim_ = model_channels * 4;
//
//     SimplifiedUNetImpl(int64_t in_c = 4, int64_t out_c = 4) {
//         in_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_c, model_channels, 3).padding(1));
//         time_proj1 = torch::nn::Linear(model_channels, time_embed_dim_);
//         time_proj2 = torch::nn::Linear(time_embed_dim_, time_embed_dim_);
//
//         down_res1 = ResBlock(model_channels, model_channels);
//         down_attn1 = AttentionBlock(model_channels);
//         down_res2 = ResBlock(model_channels, model_channels * 2);
//         downsampler1 = Downsample(model_channels * 2);
//
//         mid_res1 = ResBlock(model_channels * 2, model_channels * 2);
//         mid_attn = AttentionBlock(model_channels * 2);
//
//         upsampler1 = Upsample(model_channels * 2);
//         // After upsampler1 (ch: model_channels*2) + skip from down_res2 (ch: model_channels*2) = model_channels*4
//         up_res1 = ResBlock(model_channels * 4, model_channels);
//         up_attn1 = AttentionBlock(model_channels);
//         // After up_attn1 (ch: model_channels) + skip from down_res1 (ch: model_channels) = model_channels*2
//         up_res2 = ResBlock(model_channels * 2, model_channels);
//
//         out_norm = torch::nn::GroupNorm(torch::nn::GroupNormOptions(32, model_channels));
//         out_silu = torch::nn::SiLU();
//         out_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(model_channels, out_c, 3).padding(1));
//
//         register_module("in_conv", in_conv);
//         register_module("time_proj1", time_proj1);
//         register_module("time_proj2", time_proj2);
//         register_module("down_res1", down_res1);
//         register_module("down_attn1", down_attn1);
//         register_module("down_res2", down_res2);
//         register_module("downsampler1", downsampler1);
//         register_module("mid_res1", mid_res1);
//         register_module("mid_attn", mid_attn);
//         register_module("upsampler1", upsampler1);
//         register_module("up_res1", up_res1);
//         register_module("up_attn1", up_attn1);
//         register_module("up_res2", up_res2);
//         register_module("out_norm", out_norm);
//         register_module("out_silu", out_silu);
//         register_module("out_conv", out_conv);
//     }
//
//     torch::Tensor forward(torch::Tensor x, torch::Tensor timesteps, torch::Tensor context = torch::Tensor()) {
//         torch::Tensor t_emb = get_timestep_embedding(timesteps, model_channels);
//         t_emb = time_proj1(t_emb).silu();
//         t_emb = time_proj2(t_emb); // This would be added into ResBlocks, simplified here.
//
//         std::vector<torch::Tensor> skips;
//         x = in_conv(x); skips.push_back(x); // s0: model_channels
//
//         x = down_res1(x /* + t_emb_proj */); skips.push_back(x); // s1: model_channels
//         x = down_attn1(x, context);
//         x = down_res2(x /* + t_emb_proj */); skips.push_back(x); // s2: model_channels * 2
//         x = downsampler1(x); // model_channels * 2
//
//         x = mid_res1(x /* + t_emb_proj */);
//         x = mid_attn(x, context);
//
//         x = upsampler1(x); // model_channels * 2
//         x = torch::cat({x, skips.back()}, 1); skips.pop_back(); // s2 -> concat with upsampled, now model_channels*4
//         x = up_res1(x /* + t_emb_proj */); // model_channels
//
//         x = torch::cat({x, skips.back()}, 1); skips.pop_back(); // s1 -> concat, now model_channels*2
//         x = up_attn1(x, context);
//         x = up_res2(x /* + t_emb_proj */); // model_channels
//
//         // x = torch::cat({x, skips.back()}, 1); skips.pop_back(); // s0 // Final skip if needed, but output conv matches up_res2 output channels
//
//         x = out_norm(x);
//         x = out_silu(x);
//         return out_conv(x);
//     }
// };
// TORCH_MODULE(SimplifiedUNet);


// #pragma once
// #include "common_modules.h"
//
// struct SimplifiedVAEDecoderImpl : torch::nn::Module {
//     torch::nn::Conv2d pre_conv{nullptr};
//     // A real VAE decoder has multiple ResBlocks, Upsampling, etc.
//     Upsample up1{nullptr}; ResBlock res1{nullptr};
//     Upsample up2{nullptr}; ResBlock res2{nullptr};
//     torch::nn::GroupNorm post_norm{nullptr};
//     torch::nn::SiLU post_silu{nullptr};
//     torch::nn::Conv2d post_conv{nullptr};
//
//     SimplifiedVAEDecoderImpl(int64_t latent_c = 4, int64_t img_c = 3, int64_t base_c = 64) {
//         pre_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(latent_c, base_c * 4, 3).padding(1));
//         up1 = Upsample(base_c * 4); // To base_c * 4
//         res1 = ResBlock(base_c * 4, base_c * 2); // To base_c * 2
//         up2 = Upsample(base_c * 2); // To base_c * 2
//         res2 = ResBlock(base_c * 2, base_c); // To base_c
//         post_norm = torch::nn::GroupNorm(torch::nn::GroupNormOptions(32, base_c));
//         post_silu = torch::nn::SiLU();
//         post_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(base_c, img_c, 3).padding(1));
//
//         register_module("pre_conv", pre_conv);
//         register_module("up1", up1); register_module("res1", res1);
//         register_module("up2", up2); register_module("res2", res2);
//         register_module("post_norm", post_norm);
//         register_module("post_silu", post_silu);
//         register_module("post_conv", post_conv);
//     }
//
//     torch::Tensor forward(torch::Tensor latents) {
//         torch::Tensor x = pre_conv(latents);
//         x = up1(x); x = res1(x);
//         x = up2(x); x = res2(x);
//         // For 512x512 from 64x64, need 3 upsamples (8x). We have 2 (4x).
//         // Add one more upsample conceptually or ensure input latents are for smaller images.
//         // For example, if target is 256x256 from 64x64 latent.
//         x = post_norm(x);
//         x = post_silu(x);
//         x = post_conv(x);
//         return torch::tanh(x);
//     }
// };
// TORCH_MODULE(SimplifiedVAEDecoder);



// #pragma once
// #include "common_modules.h"
// #include <map>
// #include <string>
// #include <sstream> // For string splitting
//
// // EXTREMELY SIMPLIFIED TOKENIZER (NOT BPE)
// class SimpleCppTokenizer {
// public:
//     std::map<std::string, int> vocab_;
//     int unk_token_id_ = 0; int pad_token_id_ = 1;
//     int bos_token_id_ = 2; int eos_token_id_ = 3;
//     int max_len_ = 77;
//
//     SimpleCppTokenizer() {
//         vocab_["<|startoftext|>"] = bos_token_id_; vocab_["<|endoftext|>"] = eos_token_id_;
//         vocab_["<unk>"] = unk_token_id_;       vocab_["<pad>"] = pad_token_id_;
//         vocab_["a"] = 4; vocab_["cat"] = 5; vocab_["dog"] = 6;
//         // In reality, load from vocab.json & merges.txt for BPE
//     }
//
//     std::vector<int64_t> encode(const std::string& text) {
//         std::vector<int64_t> tokens;
//         tokens.push_back(bos_token_id_);
//         std::stringstream ss(text); std::string word;
//         while (ss >> word && tokens.size() < max_len_ - 1) {
//             std::transform(word.begin(), word.end(), word.begin(), ::tolower);
//             tokens.push_back(vocab_.count(word) ? vocab_[word] : unk_token_id_);
//         }
//         tokens.push_back(eos_token_id_);
//         while(tokens.size() < max_len_) tokens.push_back(pad_token_id_);
//         return std::vector<int64_t>(tokens.begin(), tokens.begin() + max_len_);
//     }
// };
//
// // SIMPLIFIED TEXT ENCODER (LIKE CLIP TEXT MODEL)
// struct SimplifiedTextEncoderImpl : torch::nn::Module {
//     torch::nn::Embedding token_embedding_{nullptr}, position_embedding_{nullptr};
//     // Real CLIP has multiple TransformerEncoderLayers. This is a placeholder.
//     torch::nn::Linear placeholder_transformer_layer_{nullptr}; // Not a real transformer
//     int64_t vocab_s_ = 49408; int64_t embed_d_ = 128; // Smaller embed_dim for example
//     int64_t max_pos_ = 77;
//
//     SimplifiedTextEncoderImpl() {
//         token_embedding_ = torch::nn::Embedding(vocab_s_, embed_d_);
//         position_embedding_ = torch::nn::Embedding(max_pos_, embed_d_);
//         placeholder_transformer_layer_ = torch::nn::Linear(embed_d_, embed_d_); // Just a linear layer
//         register_module("token_embedding_", token_embedding_);
//         register_module("position_embedding_", position_embedding_);
//         register_module("placeholder_transformer_layer_", placeholder_transformer_layer_);
//     }
//
//     torch::Tensor forward(torch::Tensor input_ids) {
//         int64_t seq_len = input_ids.size(1);
//         torch::Tensor pos_ids = torch::arange(seq_len, input_ids.options().dtype(torch::kLong)).unsqueeze(0);
//         torch::Tensor token_embeds = token_embedding_(input_ids);
//         torch::Tensor position_embeds = position_embedding_(pos_ids);
//         torch::Tensor embeddings = token_embeds + position_embeds;
//         // Pass through actual Transformer layers here.
//         return placeholder_transformer_layer_(embeddings); // Simplified
//     }
// };
// TORCH_MODULE(SimplifiedTextEncoder);



// #pragma once
// #include <torch/torch.h>
// #include <vector>
// #include <cmath>
// #include <numeric>   // For std::iota
// #include <algorithm> // For std::reverse, std::round
//
// class DDIMSchedulerCpp {
// public:
//     int num_train_timesteps_;
//     torch::Tensor alphas_cumprod_;
//     std::vector<int64_t> timesteps_; // Using int64_t for timesteps
//     int num_inference_steps_ = 0;
//
//     DDIMSchedulerCpp(int train_steps = 1000, double beta_start = 0.00085, double beta_end = 0.012)
//         : num_train_timesteps_(train_steps) {
//         torch::Tensor betas = torch::linspace(beta_start, beta_end, num_train_timesteps_, torch::kFloat64);
//         torch::Tensor alphas = 1.0 - betas;
//         alphas_cumprod_ = torch::cumprod(alphas, 0).to(torch::kFloat32);
//     }
//
//     void set_timesteps(int inference_steps) {
//         num_inference_steps_ = inference_steps;
//         timesteps_.resize(num_inference_steps_);
//         double step_ratio = static_cast<double>(num_train_timesteps_) / num_inference_steps_;
//         for (int i = 0; i < num_inference_steps_; ++i) {
//             timesteps_[i] = static_cast<int64_t>(std::round((num_inference_steps_ - 1 - i) * step_ratio));
//         }
//     }
//
//     torch::Tensor get_alpha_prod_t(int64_t t) {
//         if (t < 0) return torch::tensor({1.0f}, alphas_cumprod_.options());
//         return alphas_cumprod_.index({torch::tensor({t}, torch::kLong)}); // Index with tensor
//     }
//
//     torch::Tensor step(torch::Tensor model_output, int64_t timestep_idx, torch::Tensor sample, double eta = 0.0) {
//         int64_t t = timesteps_[timestep_idx];
//         int64_t prev_t = (timestep_idx + 1 < timesteps_.size()) ? timesteps_[timestep_idx + 1] : -1;
//
//         torch::Tensor alpha_prod_t = get_alpha_prod_t(t);
//         torch::Tensor alpha_prod_t_prev = get_alpha_prod_t(prev_t);
//         torch::Tensor pred_epsilon = model_output;
//         torch::Tensor pred_original_sample = (sample - torch::sqrt(1.0 - alpha_prod_t) * pred_epsilon) / torch::sqrt(alpha_prod_t);
//         torch::Tensor pred_sample_direction = torch::sqrt(1.0 - alpha_prod_t_prev) * pred_epsilon;
//         torch::Tensor prev_sample = torch::sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction;
//         if (eta > 0) { /* Add noise term based on variance, simplified here */ }
//         return prev_sample;
//     }
// };
//


// #include "unet_model.h"
// #include "vae_model.h"
// #include "text_encoder_model.h"
// #include "scheduler.h"
// #include <iostream>
// #include <opencv2/opencv.hpp> // For image saving
//
// // Function to save tensor as image (from previous examples)
// void save_image_tensor_cpp_main(torch::Tensor tensor, const std::string& file_path) {
//     tensor = tensor.add(1.0).div(2.0).mul(255.0).clamp(0, 255).to(torch::kU8);
//     tensor = tensor.permute({1, 2, 0}).contiguous();
//     cv::Mat img_mat(tensor.size(0), tensor.size(1), CV_8UC3, tensor.data_ptr<uchar>());
//     cv::cvtColor(img_mat, img_mat, cv::COLOR_RGB2BGR);
//     if (!cv::imwrite(file_path, img_mat)) {
//         std::cerr << "Error: Could not save image to " << file_path << std::endl;
//     } else { std::cout << "Saved image to " << file_path << std::endl; }
// }
//
// // !!! CRITICAL MISSING PIECE: WEIGHT LOADING FROM FILE (e.g., .safetensors or .pth) !!!
// // This is a complex C++ task involving parsing binary files and mapping tensor names.
// // For this example, models will have random weights.
// // Example: void load_weights(torch::nn::Module& model, const std::string& path_to_weights_file);
//
// int main() {
//     torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
//     std::cout << "Using device: " << device << std::endl;
//
//     // --- Instantiate Models (Random Weights) ---
//     int latent_channels = 4, image_channels = 3;
//     SimplifiedUNet unet(latent_channels, latent_channels); unet->to(device).eval();
//     SimplifiedVAEDecoder vae_decoder(latent_channels, image_channels); vae_decoder->to(device).eval();
//     SimpleCppTokenizer tokenizer;
//     SimplifiedTextEncoder text_encoder; text_encoder->to(device).eval();
//     std::cout << "Models instantiated (random weights)." << std::endl;
//     // --- Load Weights Here (NOT IMPLEMENTED) ---
//
//     // --- Inference Params ---
//     std::string prompt = "a cat"; std::string neg_prompt = "";
//     int num_steps = 10; double cfg_scale = 7.5; // Fewer steps for quick test
//     int H = 128, W = 128; // Smaller for test
//     int latent_H = H / 8, latent_W = W / 8; // Assuming 8x VAE downsampling
//     double vae_scale = 0.18215;
//
//     // --- Text Embeddings ---
//     auto prompt_ids_vec = tokenizer.encode(prompt);
//     auto neg_ids_vec = tokenizer.encode(neg_prompt);
//     torch::Tensor prompt_ids = torch::tensor(prompt_ids_vec, torch::kLong).unsqueeze(0).to(device);
//     torch::Tensor neg_ids = torch::tensor(neg_ids_vec, torch::kLong).unsqueeze(0).to(device);
//     torch::Tensor text_input_ids = torch::cat({neg_ids, prompt_ids}, 0);
//     torch::Tensor text_embeds;
//     { torch::NoGradGuard no_grad; text_embeds = text_encoder->forward(text_input_ids); }
//
//     // --- Scheduler & Latents ---
//     DDIMSchedulerCpp scheduler; scheduler.set_timesteps(num_steps);
//     torch::Tensor latents = torch::randn({1, latent_channels, latent_H, latent_W}, device);
//
//     // --- Denoising Loop ---
//     std::cout << "Starting denoising loop..." << std::endl;
//     for (int i = 0; i < num_steps; ++i) {
//         int64_t ts_val = scheduler.timesteps_[i];
//         torch::Tensor t = torch::tensor({ts_val}, torch::kLong).to(device).repeat({2}); // For CFG batch
//
//         torch::Tensor latent_model_input = torch::cat({latents, latents}, 0);
//         torch::Tensor noise_pred_batch;
//         { torch::NoGradGuard no_grad; noise_pred_batch = unet->forward(latent_model_input, t, text_embeds); }
//
//         auto noise_slices = noise_pred_batch.chunk(2, 0);
//         torch::Tensor noise_pred = noise_slices[0] + cfg_scale * (noise_slices[1] - noise_slices[0]);
//         latents = scheduler.step(noise_pred, i, latents);
//         std::cout << "Step " << i + 1 << "/" << num_steps << " (ts " << ts_val << ")" << std::endl;
//     }
//
//     // --- Decode & Save ---
//     latents = latents / vae_scale;
//     torch::Tensor image_tensor;
//     { torch::NoGradGuard no_grad; image_tensor = vae_decoder->forward(latents); }
//     save_image_tensor_cpp_main(image_tensor.squeeze(0).detach().cpu(), "sd_cpp_scratch_out.png");
//     std::cout << "C++ 'from scratch' (conceptual) SD finished (output is random noise)." << std::endl;
//     return 0;
// }

//GROK

// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // Simplified CLIP Text Encoder
// struct CLIPTextEncoderImpl : torch::nn::Module {
//     CLIPTextEncoderImpl(int text_dim = 256, int text_seq_len = 16) {
//         text_embed = register_module("text_embed", torch::nn::Embedding(10000, text_dim));
//         text_transformer = register_module("text_transformer", torch::nn::TransformerEncoder(
//             torch::nn::TransformerEncoderOptions(
//                 torch::nn::TransformerEncoderLayerOptions(text_dim, 4, text_dim * 4).dropout(0.1), 2
//             )
//         ));
//         text_pos_embed = register_parameter("text_pos_embed", torch::randn({text_seq_len, text_dim}));
//         norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({text_dim})));
//     }
//
//     torch::Tensor forward(torch::Tensor text) {
//         auto emb = text_embed->forward(text) + text_pos_embed; // [batch, text_seq_len, text_dim]
//         emb = text_transformer->forward(emb); // [batch, text_seq_len, text_dim]
//         return norm->forward(emb.mean(1)); // [batch, text_dim]
//     }
//
//     torch::nn::Embedding text_embed{nullptr};
//     torch::nn::TransformerEncoder text_transformer{nullptr};
//     torch::Tensor text_pos_embed;
//     torch::nn::LayerNorm norm{nullptr};
// };
// TORCH_MODULE(CLIPTextEncoder);
//
// // Simplified Variational Autoencoder (VAE)
// struct VAEImpl : torch::nn::Module {
//     VAEImpl() {
//         // Encoder
//         enc_conv1 = register_module("enc_conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(1, 32, 4).stride(2).padding(1)));
//         enc_conv2 = register_module("enc_conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(32, 64, 4).stride(2).padding(1)));
//         enc_fc_mu = register_module("enc_fc_mu", torch::nn::Linear(64 * 7 * 7, 128));
//         enc_fc_logvar = register_module("enc_fc_logvar", torch::nn::Linear(64 * 7 * 7, 128));
//
//         // Decoder
//         dec_fc = register_module("dec_fc", torch::nn::Linear(128, 64 * 7 * 7));
//         dec_conv1 = register_module("dec_conv1", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(64, 32, 4).stride(2).padding(1).output_padding(1)));
//         dec_conv2 = register_module("dec_conv2", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(32, 1, 4).stride(2).padding(1).output_padding(1)));
//
//         relu = register_module("relu", torch::nn::ReLU());
//         sigmoid = register_module("sigmoid", torch::nn::Sigmoid());
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
//         // Encoder
//         x = relu->forward(enc_conv1->forward(x)); // [batch, 32, 14, 14]
//         x = relu->forward(enc_conv2->forward(x)); // [batch, 64, 7, 7]
//         x = x.view({-1, 64 * 7 * 7});
//         auto mu = enc_fc_mu->forward(x); // [batch, 128]
//         auto logvar = enc_fc_logvar->forward(x); // [batch, 128]
//
//         // Reparameterization
//         auto std = torch::exp(0.5 * logvar);
//         auto eps = torch::randn_like(std);
//         auto z = mu + eps * std; // [batch, 128]
//
//         // Decoder
//         x = relu->forward(dec_fc->forward(z)); // [batch, 64 * 7 * 7]
//         x = x.view({-1, 64, 7, 7});
//         x = relu->forward(dec_conv1->forward(x)); // [batch, 32, 14, 14]
//         x = sigmoid->forward(dec_conv2->forward(x)); // [batch, 1, 28, 28]
//
//         return {x, mu, logvar};
//     }
//
//     torch::Tensor decode(torch::Tensor z) {
//         auto x = relu->forward(dec_fc->forward(z)); // [batch, 64 * 7 * 7]
//         x = x.view({-1, 64, 7, 7});
//         x = relu->forward(dec_conv1->forward(x)); // [batch, 32, 14, 14]
//         return sigmoid->forward(dec_conv2->forward(x)); // [batch, 1, 28, 28]
//     }
//
//     torch::nn::Conv2d enc_conv1{nullptr}, enc_conv2{nullptr};
//     torch::nn::Linear enc_fc_mu{nullptr}, enc_fc_logvar{nullptr};
//     torch::nn::Linear dec_fc{nullptr};
//     torch::nn::ConvTranspose2d dec_conv1{nullptr}, dec_conv2{nullptr};
//     torch::nn::ReLU relu{nullptr};
//     torch::nn::Sigmoid sigmoid{nullptr};
// };
// TORCH_MODULE(VAE);
//
// // Simplified U-Net for Latent Diffusion
// struct UNetImpl : torch::nn::Module {
//     UNetImpl(int dim = 64, int cond_dim = 256) {
//         init_conv = register_module("init_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(4, dim, 3).padding(1))); // 4 channels for latent
//         down1 = register_module("down1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(dim, dim * 2, 4).stride(2).padding(1)));
//         down2 = register_module("down2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(dim * 2, dim * 4, 4).stride(2).padding(1)));
//         mid = register_module("mid", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(dim * 4, dim * 4, 3).padding(1)));
//         up1 = register_module("up1", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(dim * 4, dim * 2, 4).stride(2).padding(1).output_padding(1)));
//         up2 = register_module("up2", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(dim * 2, dim, 4).stride(2).padding(1).output_padding(1)));
//         final_conv = register_module("final_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(dim, 4, 3).padding(1)));
//         time_embed = register_module("time_embed", torch::nn::Linear(1000, dim * 4));
//         cond_embed = register_module("cond_embed", torch::nn::Linear(cond_dim, dim * 4));
//         relu = register_module("relu", torch::nn::ReLU());
//     }
//
//     torch::Tensor forward(torch::Tensor x, torch::Tensor t, torch::Tensor cond) {
//         auto time_emb = relu->forward(time_embed->forward(t)).view({-1, 64 * 4, 1, 1}); // [batch, dim*4, 1, 1]
//         auto cond_emb = relu->forward(cond_embed->forward(cond)).view({-1, 64 * 4, 1, 1}); // [batch, dim*4, 1, 1]
//         x = relu->forward(init_conv->forward(x)); // [batch, dim, 7, 7]
//         auto h1 = relu->forward(down1->forward(x)); // [batch, dim*2, 4, 4]
//         auto h2 = relu->forward(down2->forward(h1)); // [batch, dim*4, 2, 2]
//         auto h3 = relu->forward(mid->forward(h2)) + time_emb + cond_emb; // [batch, dim*4, 2, 2]
//         auto h4 = relu->forward(up1->forward(h3)); // [batch, dim*2, 4, 4]
//         auto h5 = relu->forward(up2->forward(h4)); // [batch, dim, 7, 7]
//         return final_conv->forward(h5); // [batch, 4, 7, 7]
//     }
//
//     torch::nn::Conv2d init_conv{nullptr}, down1{nullptr}, down2{nullptr}, mid{nullptr};
//     torch::nn::ConvTranspose2d up1{nullptr}, up2{nullptr}, final_conv{nullptr};
//     torch::nn::Linear time_embed{nullptr}, cond_embed{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(UNet);
//
// // Diffusion Model
// struct DiffusionModelImpl : torch::nn::Module {
//     DiffusionModelImpl(int dim = 64, int cond_dim = 256, int timesteps = 1000) : timesteps_(timesteps) {
//         unet = register_module("unet", UNet(dim, cond_dim));
//     }
//
//     torch::Tensor forward(torch::Tensor x, torch::Tensor t, torch::Tensor cond, torch::Tensor cond_drop = torch::zeros({})) {
//         if (!cond_drop.empty()) { // Classifier-free guidance
//             auto null_cond = torch::zeros_like(cond);
//             auto cond_output = unet->forward(x, t, cond);
//             auto null_output = unet->forward(x, t, null_cond);
//             return null_output + 1.5 * (cond_output - null_output); // Guidance scale = 1.5
//         }
//         return unet->forward(x, t, cond);
//     }
//
//     torch::Tensor sample(torch::Tensor cond, torch::Device device) {
//         torch::NoGradGuard no_grad;
//         auto x = torch::randn({1, 4, 7, 7}, device); // Latent shape
//         for (int t = timesteps_ - 1; t >= 0; --t) {
//             auto t_tensor = torch::full({1}, t, torch::kInt64, device);
//             x = forward(x, t_tensor, cond);
//             if (t > 0) {
//                 x = x + torch::randn_like(x) * 0.1; // Simplified noise schedule
//             }
//         }
//         return x;
//     }
//
//     int timesteps_;
//     UNet unet{nullptr};
// };
// TORCH_MODULE(DiffusionModel);
//
// // Custom Dataset for Grayscale Images and Text
// struct TextImageDataset : torch::data::Dataset<TextImageDataset> {
//     TextImageDataset(const std::string& img_dir, const std::vector<std::string>& texts)
//         : texts_(texts) {
//         for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
//             if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
//                 image_paths_.push_back(entry.path().string());
//             }
//         }
//     }
//
//     torch::data::Example<> get(size_t index) override {
//         cv::Mat image = cv::imread(image_paths_[index % image_paths_.size()], cv::IMREAD_GRAYSCALE);
//         if (image.empty()) {
//             throw std::runtime_error("Failed to load image: " + image_paths_[index % image_paths_.size()]);
//         }
//         image.convertTo(image, CV_32F, 1.0 / 255.0);
//         torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols}, torch::kFloat32);
//         torch::Tensor text_tensor = torch::randint(0, 10000, {16}, torch::kInt64); // Mock text
//         return {img_tensor, text_tensor};
//     }
//
//     torch::optional<size_t> size() const override {
//         return image_paths_.size();
//     }
//
//     std::vector<std::string> image_paths_, texts_;
// };
//
// // Diffusion utilities
// struct DiffusionUtils {
//     DiffusionUtils(int timesteps) : timesteps_(timesteps) {
//         betas = torch::linspace(1e-4, 0.02, timesteps).to(torch::kFloat);
//         alphas = 1.0 - betas;
//         alphas_cumprod = torch::cumprod(alphas, 0);
//         sqrt_alphas_cumprod = torch::sqrt(alphas_cumprod);
//         sqrt_one_minus_alphas_cumprod = torch::sqrt(1.0 - alphas_cumprod);
//     }
//
//     torch::Tensor add_noise(torch::Tensor x, torch::Tensor t) {
//         auto sqrt_alpha = sqrt_alphas_cumprod.index({t}).view({-1, 1, 1, 1});
//         auto sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod.index({t}).view({-1, 1, 1, 1});
//         auto noise = torch::randn_like(x);
//         return sqrt_alpha * x + sqrt_one_minus_alpha * noise;
//     }
//
//     torch::Tensor sample_timesteps(int batch_size) {
//         return torch::randint(0, timesteps_, {batch_size}, torch::kInt64);
//     }
//
//     int timesteps_;
//     torch::Tensor betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod;
// };
//
// // Loss function
// torch::Tensor vae_loss(torch::Tensor recon, torch::Tensor x, torch::Tensor mu, torch::Tensor logvar) {
//     auto recon_loss = torch::nn::functional::mse_loss(recon, x);
//     auto kl_div = -0.5 * torch::sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0);
//     return recon_loss + 0.1 * kl_div;
// }
//
// int main() {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Initialize models
//         CLIPTextEncoder clip(256, 16);
//         VAE vae;
//         DiffusionModel diffusion(64, 256, 1000);
//         clip->to(device);
//         vae->to(device);
//         diffusion->to(device);
//
//         // Optimizers
//         torch::optim::Adam clip_optimizer(clip->parameters(), torch::optim::AdamOptions(0.001));
//         torch::optim::Adam vae_optimizer(vae->parameters(), torch::optim::AdamOptions(0.001));
//         torch::optim::Adam diffusion_optimizer(diffusion->parameters(), torch::optim::AdamOptions(0.001));
//
//         // Diffusion utilities
//         DiffusionUtils diffusion_utils(1000);
//
//         // Load dataset
//         std::vector<std::string> mock_texts = {"digit", "number", "image"};
//         auto dataset = TextImageDataset("./data/images", mock_texts)
//             .map(torch::data::transforms::Stack<>());
//         auto data_loader = torch::data::make_data_loader(
//             dataset, torch::data::DataLoaderOptions().batch_size(32).workers(2));
//
//         // Training loop
//         clip->train();
//         vae->train();
//         diffusion->train();
//         for (int epoch = 0; epoch < 20; ++epoch) {
//             float total_loss = 0.0;
//             int batch_count = 0;
//
//             for (auto& batch : *data_loader) {
//                 auto images = batch.data.to(device);
//                 auto text = batch.target.to(device, torch::kInt64);
//                 auto t = diffusion_utils.sample_timesteps(images.size(0)).to(device);
//                 auto cond_drop = torch::rand({images.size(0)}) < 0.1; // 10% dropout for guidance
//
//                 // VAE
//                 vae_optimizer.zero_grad();
//                 auto [recon, mu, logvar] = vae->forward(images);
//                 auto vae_loss_value = vae_loss(recon, images, mu, logvar);
//                 vae_loss_value.backward();
//                 vae_optimizer.step();
//
//                 // Get latent representation
//                 auto latent = mu.view({-1, 4, 7, 7}); // Reshape to [batch, 4, 7, 7]
//
//                 // CLIP
//                 clip_optimizer.zero_grad();
//                 auto text_emb = clip->forward(text);
//                 // No explicit CLIP loss (simplified; real model uses contrastive loss)
//
//                 // Diffusion
//                 diffusion_optimizer.zero_grad();
//                 auto noisy_latent = diffusion_utils.add_noise(latent, t);
//                 auto pred_latent = diffusion->forward(noisy_latent, t, text_emb, cond_drop.to(device));
//                 auto diffusion_loss = torch::nn::functional::mse_loss(pred_latent, latent);
//                 diffusion_loss.backward();
//                 diffusion_optimizer.step();
//
//                 total_loss += (vae_loss_value + diffusion_loss).item<float>();
//                 batch_count++;
//             }
//
//             std::cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss / batch_count << std::endl;
//         }
//
//         // Save models
//         torch::save(clip, "clip_sd.pt");
//         torch::save(vae, "vae_sd.pt");
//         torch::save(diffusion, "diffusion_sd.pt");
//         std::cout << "Models saved as clip_sd.pt, vae_sd.pt, and diffusion_sd.pt" << std::endl;
//
//         // Inference example
//         clip->eval();
//         vae->eval();
//         diffusion->eval();
//         torch::Tensor text_input = torch::randint(0, 10000, {1, 16}, torch::kInt64).to(device);
//         auto text_emb = clip->forward(text_input);
//         auto latent = diffusion->sample(text_emb, device);
//         auto generated = vae->decode(latent.view({-1, 128}));
//         generated = generated.squeeze().to(torch::kCPU);
//         cv::Mat output(28, 28, CV_32F, generated.data_ptr<float>());
//         output.convertTo(output, CV_8U, 255.0);
//         cv::imwrite("generated_sd_image.jpg", output);
//         std::cout << "Generated image saved as generated_sd_image.jpg" << std::endl;
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
    StableDiffusion::StableDiffusion(int num_classes, int in_channels)
    {
    }

    StableDiffusion::StableDiffusion(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void StableDiffusion::reset()
    {
    }

    auto StableDiffusion::forward(std::initializer_list<std::any> tensors) -> std::any
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
