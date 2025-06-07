#include "include/models/generative_models/diffusion/dall_e.h"


using namespace std;

//DALEV1

// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // Discrete Variational Autoencoder (dVAE)
// struct DiscreteVAEImpl : torch::nn::Module {
//     DiscreteVAEImpl(int num_tokens = 512, int codebook_dim = 256, int hidden_dim = 64)
//         : num_tokens_(num_tokens) {
//         // Encoder
//         enc_conv1 = register_module("enc_conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(1, 32, 4).stride(2).padding(1)));
//         enc_conv2 = register_module("enc_conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(32, 64, 4).stride(2).padding(1)));
//         enc_fc = register_module("enc_fc", torch::nn::Linear(64 * 7 * 7, num_tokens));
//
//         // Decoder
//         dec_fc = register_module("dec_fc", torch::nn::Linear(num_tokens, 64 * 7 * 7));
//         dec_conv1 = register_module("dec_conv1", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(64, 32, 4).stride(2).padding(1).output_padding(1)));
//         dec_conv2 = register_module("dec_conv2", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(32, 1, 4).stride(2).padding(1).output_padding(1)));
//
//         relu = register_module("relu", torch::nn::ReLU());
//         sigmoid = register_module("sigmoid", torch::nn::Sigmoid());
//     }
//
//     // Forward pass: returns reconstructed image and token logits
//     std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
//         // Encoder
//         x = relu->forward(enc_conv1->forward(x)); // [batch, 32, 14, 14]
//         x = relu->forward(enc_conv2->forward(x)); // [batch, 64, 7, 7]
//         x = x.view({-1, 64 * 7 * 7});
//         auto logits = enc_fc->forward(x); // [batch, num_tokens]
//
//         // Sample tokens (argmax for simplicity)
//         auto tokens = torch::argmax(logits, -1); // [batch]
//         auto token_embeds = torch::nn::functional::one_hot(tokens, num_tokens_).to(torch::kFloat);
//
//         // Decoder
//         x = relu->forward(dec_fc-> ~
//
// System: forward(token_embeds)); // [batch, 64 * 7 * 7]
//         x = x.view({-1, 64, 7, 7});
//         x = relu->forward(dec_conv1->forward(x)); // [batch, 32, 14, 14]
//         x = sigmoid->forward(dec_conv2->forward(x)); // [batch, 1, 28, 28]
//
//         return {x, logits};
//     }
//
//     int num_tokens_;
//     torch::nn::Conv2d enc_conv1{nullptr}, enc_conv2{nullptr};
//     torch::nn::Linear enc_fc{nullptr};
//     torch::nn::Linear dec_fc{nullptr};
//     torch::nn::ConvTranspose2d dec_conv1{nullptr}, dec_conv2{nullptr};
//     torch::nn::ReLU relu{nullptr};
//     torch::nn::Sigmoid sigmoid{nullptr};
// };
// TORCH_MODULE(DiscreteVAE);
//
// // Simple Transformer for Text-to-Image
// struct SimpleTransformerImpl : torch::nn::Module {
//     SimpleTransformerImpl(int dim = 256, int num_tokens = 512, int text_seq_len = 16, int depth = 4, int heads = 4)
//         : dim_(dim), num_tokens_(num_tokens), text_seq_len_(text_seq_len) {
//         // Token embeddings
//         text_embed = register_module("text_embed", torch::nn::Embedding(10000, dim));
//         image_token_embed = register_module("image_token_embed", torch::nn::Embedding(num_tokens, dim));
//
//         // Positional encodings
//         text_pos_embed = register_parameter("text_pos_embed", torch::randn({text_seq_len, dim}));
//         image_pos_embed = register_parameter("image_pos_embed", torch::randn({49, dim})); // 7x7 tokens
//
//         // Transformer layers
//         for (int i = 0; i < depth; ++i) {
//             transformer_layers.push_back(register_module(
//                 "transformer_" + std::to_string(i),
//                 torch::nn::TransformerEncoderLayer(
//                     torch::nn::TransformerEncoderLayerOptions(dim, heads, dim * 4).dropout(0.1)
//                 )
//             ));
//         }
//
//         // Output head
//         to_logits = register_module("to_logits", torch::nn::Linear(dim, num_tokens));
//     }
//
//     torch::Tensor forward(torch::Tensor text_tokens, torch::Tensor image_tokens) {
//         // Embed text and image tokens
//         auto text_emb = text_embed->forward(text_tokens) + text_pos_embed; // [batch, text_seq_len, dim]
//         auto img_emb = image_token_embed->forward(image_tokens) + image_pos_embed; // [batch, 49, dim]
//
//         // Concatenate text and image embeddings
//         auto x = torch::cat({text_emb, img_emb}, 1); // [batch, text_seq_len + 49, dim]
//
//         // Transformer layers
//         for (auto& layer : transformer_layers) {
//             x = layer->forward(x);
//         }
//
//         // Predict next image token logits
//         x = to_logits->forward(x[:, text_seq_len_]); // [batch, 49, num_tokens]
//         return x;
//     }
//
//     int dim_, num_tokens_, text_seq_len_;
//     torch::nn::Embedding text_embed{nullptr}, image_token_embed{nullptr};
//     torch::Tensor text_pos_embed, image_pos_embed;
//     std::vector<torch::nn::TransformerEncoderLayer> transformer_layers;
//     torch::nn::Linear to_logits{nullptr};
// };
// TORCH_MODULE(SimpleTransformer);
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
//         // Load image
//         cv::Mat image = cv::imread(image_paths_[index % image_paths_.size()], cv::IMREAD_GRAYSCALE);
//         if (image.empty()) {
//             throw std::runtime_error("Failed to load image: " + image_paths_[index % image_paths_.size()]);
//         }
//         image.convertTo(image, CV_32F, 1.0 / 255.0);
//         torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols}, torch::kFloat32);
//
//         // Mock text tokens (simplified: random tokens for demo)
//         torch::Tensor text_tensor = torch::randint(0, 10000, {16}, torch::kInt64);
//
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
// // Loss function: Reconstruction + Cross-Entropy for transformer
// struct DALLELoss {
//     torch::Tensor operator()(torch::Tensor recon, torch::Tensor target, torch::Tensor token_logits, torch::Tensor target_tokens) {
//         auto recon_loss = torch::nn::functional::mse_loss(recon, target);
//         auto ce_loss = torch::nn::functional::cross_entropy(token_logits.view({-1, token_logits.size(-1)}), target_tokens.view(-1));
//         return recon_loss + ce_loss;
//     }
// };
//
// int main() {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Initialize models
//         DiscreteVAE vae(512, 256, 64); // 512 tokens
//         SimpleTransformer transformer(256, 512, 16, 4, 4); // dim=256, depth=4, heads=4
//         vae->to(device);
//         transformer->to(device);
//
//         // Optimizers
//         torch::optim::Adam vae_optimizer(vae->parameters(), torch::optim::AdamOptions(0.001));
//         torch::optim::Adam transformer_optimizer(transformer->parameters(), torch::optim::AdamOptions(0.001));
//
//         // Load dataset
//         std::vector<std::string> mock_texts = {"digit", "number", "image"}; // Placeholder
//         auto dataset = TextImageDataset("./data/images", mock_texts)
//             .map(torch::data::transforms::Stack<>());
//         auto data_loader = torch::data::make_data_loader(
//             dataset, torch::data::DataLoaderOptions().batch_size(32).workers(2));
//
//         // Training loop
//         vae->train();
//         transformer->train();
//         for (int epoch = 0; epoch < 20; ++epoch) {
//             float total_loss = 0.0;
//             int batch_count = 0;
//
//             for (auto& batch : *data_loader) {
//                 auto images = batch.data.to(device);
//                 auto text = batch.target.to(device, torch::kInt64);
//
//                 // Train VAE
//                 vae_optimizer.zero_grad();
//                 auto [recon, logits] = vae->forward(images);
//                 auto target_tokens = torch::argmax(logits, -1); // [batch, num_tokens]
//                 auto vae_loss = torch::nn::functional::mse_loss(recon, images);
//                 vae_loss.backward();
//                 vae_optimizer.step();
//
//                 // Train Transformer
//                 transformer_optimizer.zero_grad();
//                 auto transformer_logits = transformer->forward(text, target_tokens);
//                 auto transformer_loss = torch::nn::functional::cross_entropy(
//                     transformer_logits.view({-1, 512}), target_tokens.view(-1));
//                 transformer_loss.backward();
//                 transformer_optimizer.step();
//
//                 total_loss += (vae_loss + transformer_loss).item<float>();
//                 batch_count++;
//             }
//
//             std::cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss / batch_count << std::endl;
//         }
//
//         // Save models
//         torch::save(vae, "dvae.pt");
//         torch::save(transformer, "transformer.pt");
//         std::cout << "Models saved as dvae.pt and transformer.pt" << std::endl;
//
//         // Inference example
//         vae->eval();
//         transformer->eval();
//         cv::Mat test_image = cv::imread("test_image.jpg", cv::IMREAD_GRAYSCALE);
//         if (test_image.empty()) {
//             std::cerr << "Error: Could not load test image." << std::endl;
//             return -1;
//         }
//         test_image.convertTo(test_image, CV_32F, 1.0 / 255.0);
//         torch::Tensor test_tensor = torch::from_blob(
//             test_image.data, {1, 1, test_image.rows, test_image.cols}, torch::kFloat32
//         ).to(device);
//
//         // Mock text input
//         torch::Tensor text_input = torch::randint(0, 10000, {1, 16}, torch::kInt64).to(device);
//
//         // Generate image tokens
//         auto [_, logits] = vae->forward(test_tensor);
//         auto tokens = torch::argmax(logits, -1);
//         auto transformer_logits = transformer->forward(text_input, tokens);
//         auto gen_tokens = torch::argmax(transformer_logits, -1);
//
//         // Decode tokens to image
//         auto token_embeds = torch::nn::functional::one_hot(gen_tokens, 512).to(torch::kFloat);
//         auto recon = vae->forward(token_embeds).first.squeeze().to(torch::kCPU);
//         cv::Mat reconstructed(
//             test_image.rows, test_image.cols, CV_32F, recon.data_ptr<float>()
//         );
//         reconstructed.convertTo(reconstructed, CV_8U, 255.0);
//         cv::imwrite("reconstructed_dalle_image.jpg", reconstructed);
//         std::cout << "Generated image saved as reconstructed_dalle_image.jpg" << std::endl;
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
//
//     return 0;
// }



//DallEV2


//
// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // Simplified CLIP Model (Text and Image Encoder)
// struct CLIPImpl : torch::nn::Module {
//     CLIPImpl(int text_dim = 256, int image_dim = 256, int latent_dim = 256, int text_seq_len = 16) {
//         text_embed = register_module("text_embed", torch::nn::Embedding(10000, text_dim));
//         text_transformer = register_module("text_transformer", torch::nn::TransformerEncoder(
//             torch::nn::TransformerEncoderOptions(
//                 torch::nn::TransformerEncoderLayerOptions(text_dim, 4, text_dim * 4).dropout(0.1), 2
//             )
//         ));
//         text_pos_embed = register_parameter("text_pos_embed", torch::randn({text_seq_len, text_dim}));
//         image_conv1 = register_module("image_conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(1, 32, 4).stride(2).padding(1)));
//         image_conv2 = register_module("image_conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(32, 64, 4).stride(2).padding(1)));
//         image_fc = register_module("image_fc", torch::nn::Linear(64 * 7 * 7, image_dim));
//         to_latent = register_module("to_latent", torch::nn::Linear(text_dim + image_dim, latent_dim));
//         relu = register_module("relu", torch::nn::ReLU());
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor text, torch::Tensor image) {
//         // Text encoding
//         auto text_emb = text_embed->forward(text) + text_pos_embed; // [batch, text_seq_len, text_dim]
//         text_emb = text_transformer->forward(text_emb).mean(1); // [batch, text_dim]
//
//         // Image encoding
//         auto img_emb = relu->forward(image_conv1->forward(image)); // [batch, 32, 14, 14]
//         img_emb = relu->forward(image_conv2->forward(img_emb)); // [batch, 64, 7, 7]
//         img_emb = img_emb.view({-1, 64 * 7 * 7});
//         img_emb = image_fc->forward(img_emb); // [batch, image_dim]
//
//         // Combine
//         auto combined = torch::cat({text_emb, img_emb}, -1); // [batch, text_dim + image_dim]
//         auto latent = to_latent->forward(combined); // [batch, latent_dim]
//         return {text_emb, img_emb};
//     }
//
//     torch::nn::Embedding text_embed{nullptr};
//     torch::nn::TransformerEncoder text_transformer{nullptr};
//     torch::Tensor text_pos_embed;
//     torch::nn::Conv2d image_conv1{nullptr}, image_conv2{nullptr};
//     torch::nn::Linear image_fc{nullptr}, to_latent{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(CLIP);
//
// // Simplified Diffusion Prior (Predicts Image Embedding from Text Embedding)
// struct DiffusionPriorImpl : torch::nn::Module {
//     DiffusionPriorImpl(int dim = 256, int timesteps = 1000) : timesteps_(timesteps) {
//         to_embedding = register_module("to_embedding", torch::nn::Linear(dim, dim));
//         time_embed = register_module("time_embed", torch::nn::Embedding(timesteps, dim));
//         transformer = register_module("transformer", torch::nn::TransformerEncoder(
//             torch::nn::TransformerEncoderOptions(
//                 torch::nn::TransformerEncoderLayerOptions(dim, 4, dim * 4).dropout(0.1), 2
//             )
//         ));
//         to_output = register_module("to_output", torch::nn::Linear(dim, dim));
//         relu = register_module("relu", torch::nn::ReLU());
//     }
//
//     torch::Tensor forward(torch::Tensor text_emb, torch::Tensor t, torch::Tensor img_emb_noisy) {
//         auto time_emb = time_embed->forward(t); // [batch, dim]
//         auto x = to_embedding->forward(text_emb + time_emb); // [batch, dim]
//         x = x.unsqueeze(1); // [batch, 1, dim]
//         x = transformer->forward(x).squeeze(1); // [batch, dim]
//         return to_output->forward(x); // [batch, dim]
//     }
//
//     int timesteps_;
//     torch::nn::Linear to_embedding{nullptr}, to_output{nullptr};
//     torch::nn::Embedding time_embed{nullptr};
//     torch::nn::TransformerEncoder transformer{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(DiffusionPrior);
//
// // Simplified Diffusion Decoder
// struct DiffusionDecoderImpl : torch::nn::Module {
//     DiffusionDecoderImpl(int dim = 256, int timesteps = 1000) : timesteps_(timesteps) {
//         time_embed = register_module("time_embed", torch::nn::Embedding(timesteps, dim));
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(1 + dim / (7 * 7), 64, 3).padding(1)));
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 32, 3).padding(1)));
//         conv3 = register_module("conv3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(32, 1, 3).padding(1)));
//         relu = register_module("relu", torch::nn::ReLU());
//         sigmoid = register_module("sigmoid", torch::nn::Sigmoid());
//     }
//
//     torch::Tensor forward(torch::Tensor x, torch::Tensor t, torch::Tensor cond) {
//         auto time_emb = time_embed->forward(t); // [batch, dim]
//         auto cond_reshaped = cond.view({-1, dim / (7 * 7), 7, 7}); // [batch, dim/(7*7), 7, 7]
//         x = torch::cat({x, cond_reshaped}, 1); // [batch, 1 + dim/(7*7), 7, 7]
//         x = relu->forward(conv1->forward(x)); // [batch, 64, 7, 7]
//         x = relu->forward(conv2->forward(x)); // [batch, 32, 7, 7]
//         x = sigmoid->forward(conv3->forward(x)); // [batch, 1, 7, 7]
//         x = torch::nn::functional::interpolate(x, {28, 28}, torch::nn::functional::InterpolateFuncOptions().mode(torch::kBilinear).align_corners(false));
//         return x;
//     }
//
//     int timesteps_;
//     torch::nn::Embedding time_embed{nullptr};
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
//     torch::nn::ReLU relu{nullptr};
//     torch::nn::Sigmoid sigmoid{nullptr};
// };
// TORCH_MODULE(DiffusionDecoder);
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
// struct DALLE2Loss {
//     torch::Tensor operator()(torch::Tensor clip_loss, torch::Tensor prior_loss, torch::Tensor decoder_loss) {
//         return clip_loss + prior_loss + decoder_loss;
//     }
// };
//
// int main() {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Initialize models
//         CLIP clip(256, 256, 256, 16);
//         DiffusionPrior prior(256, 1000);
//         DiffusionDecoder decoder(256, 1000);
//         clip->to(device);
//         prior->to(device);
//         decoder->to(device);
//
//         // Optimizers
//         torch::optim::Adam clip_optimizer(clip->parameters(), torch::optim::AdamOptions(0.001));
//         torch::optim::Adam prior_optimizer(prior->parameters(), torch::optim::AdamOptions(0.001));
//         torch::optim::Adam decoder_optimizer(decoder->parameters(), torch::optim::AdamOptions(0.001));
//
//         // Diffusion utilities
//         DiffusionUtils diffusion(1000);
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
//         prior->train();
//         decoder->train();
//         for (int epoch = 0; epoch < 20; ++epoch) {
//             float total_loss = 0.0;
//             int batch_count = 0;
//
//             for (auto& batch : *data_loader) {
//                 auto images = batch.data.to(device);
//                 auto text = batch.target.to(device, torch::kInt64);
//                 auto t = diffusion.sample_timesteps(images.size(0)).to(device);
//
//                 // CLIP
//                 clip_optimizer.zero_grad();
//                 auto [text_emb, img_emb] = clip->forward(text, images);
//                 auto clip_loss = torch::nn::functional::mse_loss(text_emb, img_emb);
//                 clip_loss.backward();
//                 clip_optimizer.step();
//
//                 // Prior
//                 prior_optimizer.zero_grad();
//                 auto noisy_img_emb = img_emb + torch::randn_like(img_emb) * 0.1;
//                 auto pred_img_emb = prior->forward(text_emb, t, noisy_img_emb);
//                 auto prior_loss = torch::nn::functional::mse_loss(pred_img_emb, img_emb);
//                 prior_loss.backward();
//                 prior_optimizer.step();
//
//                 // Decoder
//                 decoder_optimizer.zero_grad();
//                 auto noisy_images = diffusion.add_noise(images, t);
//                 auto recon = decoder->forward(noisy_images, t, pred_img_emb);
//                 auto decoder_loss = torch::nn::functional::mse_loss(recon, images);
//                 decoder_loss.backward();
//                 decoder_optimizer.step();
//
//                 total_loss += (clip_loss + prior_loss + decoder_loss).item<float>();
//                 batch_count++;
//             }
//
//             std::cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss / batch_count << std::endl;
//         }
//
//         // Save models
//         torch::save(clip, "clip.pt");
//         torch::save(prior, "prior.pt");
//         torch::save(decoder, "decoder.pt");
//         std::cout << "Models saved as clip.pt, prior.pt, and decoder.pt" << std::endl;
//
//         // Inference example
//         clip->eval();
//         prior->eval();
//         decoder->eval();
//         torch::Tensor text_input = torch::randint(0, 10000, {1, 16}, torch::kInt64).to(device);
//         auto [text_emb, _] = clip->forward(text_input, torch::zeros({1, 1, 28, 28}, torch::kFloat).to(device));
//         auto img_emb = text_emb;
//         for (int t = 999; t >= 0; --t) {
//             auto t_tensor = torch::full({1}, t, torch::kInt64).to(device);
//             img_emb = prior->forward(text_emb, t_tensor, img_emb);
//         }
//         torch::Tensor x = torch::randn({1, 1, 28, 28}, torch::kFloat).to(device);
//         for (int t = 999; t >= 0; --t) {
//             auto t_tensor = torch::full({1}, t, torch::kInt64).to(device);
//             x = decoder->forward(x, t_tensor, img_emb);
//         }
//         x = x.squeeze().to(torch::kCPU);
//         cv::Mat generated(28, 28, CV_32F, x.data_ptr<float>());
//         generated.convertTo(generated, CV_8U, 255.0);
//         cv::imwrite("generated_dalle2_image.jpg", generated);
//         std::cout << "Generated image saved as generated_dalle2_image.jpg" << std::endl;
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
//
//     return 0;
// }
//

//DallEV3


#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>

// Enhanced CLIP Model (Text and Image Encoder)
struct CLIPImpl : torch::nn::Module {
    CLIPImpl(int text_dim = 512, int image_dim = 512, int latent_dim = 512, int text_seq_len = 32) {
        text_embed = register_module("text_embed", torch::nn::Embedding(20000, text_dim));
        text_transformer = register_module("text_transformer", torch::nn::TransformerEncoder(
            torch::nn::TransformerEncoderOptions(
                torch::nn::TransformerEncoderLayerOptions(text_dim, 8, text_dim * 4).dropout(0.1), 4
            )
        ));
        text_pos_embed = register_parameter("text_pos_embed", torch::randn({text_seq_len, text_dim}));
        image_conv1 = register_module("image_conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1)));
        image_conv2 = register_module("image_conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1)));
        image_fc = register_module("image_fc", torch::nn::Linear(128 * 7 * 7, image_dim));
        to_latent = register_module("to_latent", torch::nn::Linear(text_dim + image_dim, latent_dim));
        relu = register_module("relu", torch::nn::ReLU());
        norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({latent_dim})));
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor text, torch::Tensor image, torch::Tensor text_mask) {
        // Text encoding
        auto text_emb = text_embed->forward(text) + text_pos_embed; // [batch, text_seq_len, text_dim]
        text_emb = text_transformer->forward(text_emb, text_mask); // [batch, text_seq_len, text_dim]
        text_emb = text_emb.mean(1); // [batch, text_dim]

        // Image encoding
        auto img_emb = relu->forward(image_conv1->forward(image)); // [batch, 64, 14, 14]
        img_emb = relu->forward(image_conv2->forward(img_emb)); // [batch, 128, 7, 7]
        img_emb = img_emb.view({-1, 128 * 7 * 7});
        img_emb = image_fc->forward(img_emb); // [batch, image_dim]

        // Combine and normalize
        auto combined = torch::cat({text_emb, img_emb}, -1); // [batch, text_dim + image_dim]
        auto latent = norm->forward(to_latent->forward(combined)); // [batch, latent_dim]
        return {text_emb, img_emb};
    }

    torch::nn::Embedding text_embed{nullptr};
    torch::nn::TransformerEncoder text_transformer{nullptr};
    torch::Tensor text_pos_embed;
    torch::nn::Conv2d image_conv1{nullptr}, image_conv2{nullptr};
    torch::nn::Linear image_fc{nullptr}, to_latent{nullptr};
    torch::nn::ReLU relu{nullptr};
    torch::nn::LayerNorm norm{nullptr};
};
TORCH_MODULE(CLIP);

// Enhanced Diffusion Prior
struct DiffusionPriorImpl : torch::nn::Module {
    DiffusionPriorImpl(int dim = 512, int timesteps = 1000) : timesteps_(timesteps) {
        to_embedding = register_module("to_embedding", torch::nn::Linear(dim, dim));
        time_embed = register_module("time_embed", torch::nn::Embedding(timesteps, dim));
        transformer = register_module("transformer", torch::nn::TransformerEncoder(
            torch::nn::TransformerEncoderOptions(
                torch::nn::TransformerEncoderLayerOptions(dim, 8, dim * 4).dropout(0.1), 4
            )
        ));
        to_output = register_module("to_output", torch::nn::Linear(dim, dim));
        relu = register_module("relu", torch::nn::ReLU());
    }

    torch::Tensor forward(torch::Tensor text_emb, torch::Tensor t, torch::Tensor img_emb_noisy, torch::Tensor cond_drop = torch::zeros({})) {
        auto time_emb = time_embed->forward(t); // [batch, dim]
        auto x = text_emb + time_emb;
        if (!cond_drop.empty()) { // Classifier-free guidance
            x = torch::where(cond_drop.unsqueeze(-1).expand_as(x), torch::zeros_like(x), x);
        }
        x = to_embedding->forward(x); // [batch, dim]
        x = x.unsqueeze(1); // [batch, 1, dim]
        x = transformer->forward(x).squeeze(1); // [batch, dim]
        return to_output->forward(x); // [batch, dim]
    }

    int timesteps_;
    torch::nn::Linear to_embedding{nullptr}, to_output{nullptr};
    torch::nn::Embedding time_embed{nullptr};
    torch::nn::TransformerEncoder transformer{nullptr};
    torch::nn::ReLU relu{nullptr};
};
TORCH_MODULE(DiffusionPrior);

// Enhanced Diffusion Decoder with U-Net
struct UNetImpl : torch::nn::Module {
    UNetImpl(int dim = 128, int cond_dim = 512) {
        init_conv = register_module("init_conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(1, dim, 3).padding(1)));
        down1 = register_module("down1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(dim, dim * 2, 4).stride(2).padding(1)));
        down2 = register_module("down2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(dim * 2, dim * 4, 4).stride(2).padding(1)));
        mid = register_module("mid", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(dim * 4 + cond_dim / (7 * 7), dim * 4, 3).padding(1)));
        up1 = register_module("up1", torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(dim * 4, dim * 2, 4).stride(2).padding(1).output_padding(1)));
        up2 = register_module("up2", torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(dim * 2, dim, 4).stride(2).padding(1).output_padding(1)));
        final_conv = register_module("final_conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(dim, 1, 3).padding(1)));
        time_embed = register_module("time_embed", torch::nn::Linear(1000, dim * 4));
        relu = register_module("relu", torch::nn::ReLU());
        sigmoid = register_module("sigmoid", torch::nn::Sigmoid());
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor t, torch::Tensor cond) {
        auto time_emb = relu->forward(time_embed->forward(t)).view({-1, 128 * 4, 1, 1}); // [batch, dim*4, 1, 1]
        x = relu->forward(init_conv->forward(x)); // [batch, dim, 28, 28]
        auto h1 = relu->forward(down1->forward(x)); // [batch, dim*2, 14, 14]
        auto h2 = relu->forward(down2->forward(h1)); // [batch, dim*4, 7, 7]
        auto cond_reshaped = cond.view({-1, cond.size(-1) / (7 * 7), 7, 7}); // [batch, cond_dim/(7*7), 7, 7]
        auto h3 = torch::cat({h2, cond_reshaped}, 1); // [batch, dim*4 + cond_dim/(7*7), 7, 7]
        h3 = relu->forward(mid->forward(h3)) + time_emb; // [batch, dim*4, 7, 7]
        auto h4 = relu->forward(up1->forward(h3)); // [batch, dim*2, 14, 14]
        auto h5 = relu->forward(up2->forward(h4)); // [batch, dim, 28, 28]
        return sigmoid->forward(final_conv->forward(h5)); // [batch, 1, 28, 28]
    }

    torch::nn::Conv2d init_conv{nullptr}, down1{nullptr}, down2{nullptr}, mid{nullptr};
    torch::nn::ConvTranspose2d up1{nullptr}, up2{nullptr}, final_conv{nullptr};
    torch::nn::Linear time_embed{nullptr};
    torch::nn::ReLU relu{nullptr};
    torch::nn::Sigmoid sigmoid{nullptr};
};
TORCH_MODULE(UNet);

struct DiffusionDecoderImpl : torch::nn::Module {
    DiffusionDecoderImpl(int dim = 128, int cond_dim = 512, int timesteps = 1000) : timesteps_(timesteps) {
        unet = register_module("unet", UNet(dim, cond_dim));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor t, torch::Tensor cond, torch::Tensor cond_drop = torch::zeros({})) {
        if (!cond_drop.empty()) { // Classifier-free guidance
            auto null_cond = torch::zeros_like(cond);
            auto cond_output = unet->forward(x, t, cond);
            auto null_output = unet->forward(x, t, null_cond);
            return null_output + 1.0 * (cond_output - null_output); // Guidance scale = 1.0
        }
        return unet->forward(x, t, cond);
    }

    torch::Tensor sample(torch::Tensor cond, torch::Device device) {
        torch::NoGradGuard no_grad;
        auto x = torch::randn({1, 1, 28, 28}, device);
        for (int t = timesteps_ - 1; t >= 0; --t) {
            auto t_tensor = torch::full({1}, t, torch::kInt64, device);
            x = forward(x, t_tensor, cond);
            if (t > 0) {
                x = x + torch::randn_like(x) * 0.1; // Simplified noise schedule
            }
        }
        return x;
    }

    int timesteps_;
    UNet unet{nullptr};
};
TORCH_MODULE(DiffusionDecoder);

// Custom Dataset for Grayscale Images and Text
struct TextImageDataset : torch::data::Dataset<TextImageDataset> {
    TextImageDataset(const std::string& img_dir, const std::vector<std::string>& texts)
        : texts_(texts) {
        for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
            if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
                image_paths_.push_back(entry.path().string());
            }
        }
    }

    torch::data::Example<> get(size_t index) override {
        cv::Mat image = cv::imread(image_paths_[index % image_paths_.size()], cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            throw std::runtime_error("Failed to load image: " + image_paths_[index % image_paths_.size()]);
        }
        image.convertTo(image, CV_32F, 1.0 / 255.0);
        torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols}, torch::kFloat32);
        torch::Tensor text_tensor = torch::randint(0, 20000, {32}, torch::kInt64); // Mock text
        torch::Tensor mask = torch::ones({32}, torch::kBool); // Mock mask
        return {img_tensor, torch::cat({text_tensor.unsqueeze(0), mask.unsqueeze(0)}, 0)};
    }

    torch::optional<size_t> size() const override {
        return image_paths_.size();
    }

    std::vector<std::string> image_paths_, texts_;
};

// Diffusion utilities
struct DiffusionUtils {
    DiffusionUtils(int timesteps) : timesteps_(timesteps) {
        betas = torch::linspace(1e-4, 0.02, timesteps).to(torch::kFloat);
        alphas = 1.0 - betas;
        alphas_cumprod = torch::cumprod(alphas, 0);
        sqrt_alphas_cumprod = torch::sqrt(alphas_cumprod);
        sqrt_one_minus_alphas_cumprod = torch::sqrt(1.0 - alphas_cumprod);
    }

    torch::Tensor add_noise(torch::Tensor x, torch::Tensor t) {
        auto sqrt_alpha = sqrt_alphas_cumprod.index({t}).view({-1, 1, 1, 1});
        auto sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod.index({t}).view({-1, 1, 1, 1});
        auto noise = torch::randn_like(x);
        return sqrt_alpha * x + sqrt_one_minus_alpha * noise;
    }

    torch::Tensor sample_timesteps(int batch_size) {
        return torch::randint(0, timesteps_, {batch_size}, torch::kInt64);
    }

    int timesteps_;
    torch::Tensor betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod;
};

// Loss function
struct DALLE3Loss {
    torch::Tensor operator()(torch::Tensor clip_loss, torch::Tensor prior_loss, torch::Tensor decoder_loss) {
        return clip_loss + prior_loss + decoder_loss;
    }
};

int main() {
    try {
        // Set device
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

        // Initialize models
        CLIP clip(512, 512, 512, 32);
        DiffusionPrior prior(512, 1000);
        DiffusionDecoder decoder(128, 512, 1000);
        clip->to(device);
        prior->to(device);
        decoder->to(device);

        // Optimizers
        torch::optim::Adam clip_optimizer(clip->parameters(), torch::optim::AdamOptions(0.0003));
        torch::optim::Adam prior_optimizer(prior->parameters(), torch::optim::AdamOptions(0.0003));
        torch::optim::Adam decoder_optimizer(decoder->parameters(), torch::optim::AdamOptions(0.0003));

        // Diffusion utilities
        DiffusionUtils diffusion(1000);

        // Load dataset
        std::vector<std::string> mock_texts = {"digit", "number", "image"};
        auto dataset = TextImageDataset("./data/images", mock_texts)
            .map(torch::data::transforms::Stack<>());
        auto data_loader = torch::data::make_data_loader(
            dataset, torch::data::DataLoaderOptions().batch_size(32).workers(2));

        // Training loop
        clip->train();
        prior->train();
        decoder->train();
        for (int epoch = 0; epoch < 20; ++epoch) {
            float total_loss = 0.0;
            int batch_count = 0;

            for (auto& batch : *data_loader) {
                auto images = batch.data.to(device);
                auto text_and_mask = batch.target.to(device);
                auto text = text_and_mask.slice(0, 0, 1).squeeze(0); // [batch, 32]
                auto mask = text_and_mask.slice(0, 1, 2).squeeze(0).to(torch::kBool); // [batch, 32]
                auto t = diffusion.sample_timesteps(images.size(0)).to(device);
                auto cond_drop = torch::rand({images.size(0)}) < 0.1; // 10% dropout for guidance

                // CLIP
                clip_optimizer.zero_grad();
                auto [text_emb, img_emb] = clip->forward(text, images, mask);
                auto clip_loss = torch::nn::functional::mse_loss(text_emb, img_emb);
                clip_loss.backward();
                clip_optimizer.step();

                // Prior
                prior_optimizer.zero_grad();
                auto noisy_img_emb = img_emb + torch::randn_like(img_emb) * 0.1;
                auto pred_img_emb = prior->forward(text_emb, t, noisy_img_emb, cond_drop.to(device));
                auto prior_loss = torch::nn::functional::mse_loss(pred_img_emb, img_emb);
                prior_loss.backward();
                prior_optimizer.step();

                // Decoder
                decoder_optimizer.zero_grad();
                auto noisy_images = diffusion.add_noise(images, t);
                auto recon = decoder->forward(noisy_images, t, pred_img_emb, cond_drop.to(device));
                auto decoder_loss = torch::nn::functional::mse_loss(recon, images);
                decoder_loss.backward();
                decoder_optimizer.step();

                total_loss += (clip_loss + prior_loss + decoder_loss).item<float>();
                batch_count++;
            }

            std::cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss / batch_count << std::endl;
        }

        // Save models
        torch::save(clip, "clip_v3.pt");
        torch::save(prior, "prior_v3.pt");
        torch::save(decoder, "decoder_v3.pt");
        std::cout << "Models saved as clip_v3.pt, prior_v3.pt, and decoder_v3.pt" << std::endl;

        // Inference example
        clip->eval();
        prior->eval();
        decoder->eval();
        torch::Tensor text_input = torch::randint(0, 20000, {1, 32}, torch::kInt64).to(device);
        torch::Tensor text_mask = torch::ones({1, 32}, torch::kBool).to(device);
        auto [text_emb, _] = clip->forward(text_input, torch::zeros({1, 1, 28, 28}, torch::kFloat).to(device), text_mask);
        auto img_emb = text_emb;
        for (int t = 999; t >= 0; --t) {
            auto t_tensor = torch::full({1}, t, torch::kInt64, device);
            img_emb = prior->forward(text_emb, t_tensor, img_emb, torch::zeros({1}, torch::kBool, device));
        }
        auto generated = decoder->sample(img_emb, device);
        generated = generated.squeeze().to(torch::kCPU);
        cv::Mat output(28, 28, CV_32F, generated.data_ptr<float>());
        output.convertTo(output, CV_8U, 255.0);
        cv::imwrite("generated_dalle3_image.jpg", output);
        std::cout << "Generated image saved as generated_dalle3_image.jpg" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}



namespace xt::models
{
    DallEV1::DallEV1(int num_classes, int in_channels)
    {
    }

    DallEV1::DallEV1(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DallEV1::reset()
    {
    }

    auto DallEV1::forward(std::initializer_list<std::any> tensors) -> std::any
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

    DallEV2::DallEV2(int num_classes, int in_channels)
    {
    }

    DallEV2::DallEV2(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DallEV2::reset()
    {
    }

    auto DallEV2::forward(std::initializer_list<std::any> tensors) -> std::any
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

    DallEV3::DallEV3(int num_classes, int in_channels)
    {
    }

    DallEV3::DallEV3(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DallEV3::reset()
    {
    }

    auto DallEV3::forward(std::initializer_list<std::any> tensors) -> std::any
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
