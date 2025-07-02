#include "include/models/generative_models/gans/pro_gan.h"


using namespace std;

//PROGAN GROK

//
// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // Generator Block
// struct GeneratorBlockImpl : torch::nn::Module {
//     GeneratorBlockImpl(int in_channels, int out_channels) {
//         conv = register_module("conv", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 4).stride(2).padding(1)));
//         bn = register_module("bn", torch::nn::BatchNorm2d(out_channels));
//         relu = register_module("relu", torch::nn::ReLU());
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = relu->forward(bn->forward(conv->forward(x)));
//         return x;
//     }
//
//     torch::nn::ConvTranspose2d conv{nullptr};
//     torch::nn::BatchNorm2d bn{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(GeneratorBlock);
//
// // Generator
// struct ProGANGeneratorImpl : torch::nn::Module {
//     ProGANGeneratorImpl(int latent_dim) : latent_dim_(latent_dim) {
//         initial = register_module("initial", torch::nn::Linear(latent_dim, 256 * 4 * 4));
//         block1 = register_module("block1", GeneratorBlock(256, 128)); // 4x4 -> 8x8
//         block2 = register_module("block2", GeneratorBlock(128, 64)); // 8x8 -> 16x16
//         block3 = register_module("block3", GeneratorBlock(64, 32)); // 16x16 -> 28x28
//         to_img_4x4 = register_module("to_img_4x4", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 1, 1).stride(1)));
//         to_img_8x8 = register_module("to_img_8x8", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(128, 1, 1).stride(1)));
//         to_img_16x16 = register_module("to_img_16x16", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 1, 1).stride(1)));
//         to_img_28x28 = register_module("to_img_28x28", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(32, 1, 3).stride(1).padding(1)));
//         relu = register_module("relu", torch::nn::ReLU());
//     }
//
//     torch::Tensor forward(torch::Tensor z, int resolution_level, float alpha) {
//         auto batch_size = z.size(0);
//         z = relu->forward(initial->forward(z)); // [batch, 256 * 4 * 4]
//         z = z.view({batch_size, 256, 4, 4}); // [batch, 256, 4, 4]
//
//         if (resolution_level == 0) {
//             return torch::tanh(to_img_4x4->forward(z)); // [batch, 1, 4, 4]
//         }
//
//         z = block1->forward(z); // [batch, 128, 8, 8]
//         if (resolution_level == 1) {
//             auto img = to_img_8x8->forward(z);
//             return torch::tanh(img);
//         } else if (resolution_level == 2) {
//             auto img_prev = torch::upsample_nearest2d(to_img_4x4->forward(block1->forward(z)), {8, 8});
//             auto img_curr = to_img_8x8->forward(z);
//             return torch::tanh((1 - alpha) * img_prev + alpha * img_curr); // Fade-in
//         }
//
//         z = block2->forward(z); // [batch, 64, 16, 16]
//         if (resolution_level == 3) {
//             auto img = to_img_16x16->forward(z);
//             return torch::tanh(img);
//         } else if (resolution_level == 4) {
//             auto img_prev = torch::upsample_nearest2d(to_img_8x8->forward(block2->forward(z)), {16, 16});
//             auto img_curr = to_img_16x16->forward(z);
//             return torch::tanh((1 - alpha) * img_prev + alpha * img_curr); // Fade-in
//         }
//
//         z = block3->forward(z); // [batch, 32, 28, 28]
//         if (resolution_level == 5) {
//             auto img = to_img_28x28->forward(z);
//             return torch::tanh(img);
//         } else if (resolution_level == 6) {
//             auto img_prev = torch::upsample_nearest2d(to_img_16x16->forward(block2->forward(z)), {28, 28});
//             auto img_curr = to_img_28x28->forward(z);
//             return torch::tanh((1 - alpha) * img_prev + alpha * img_curr); // Fade-in
//         }
//
//         throw std::runtime_error("Invalid resolution level");
//     }
//
//     int latent_dim_;
//     torch::nn::Linear initial{nullptr};
//     GeneratorBlock block1{nullptr}, block2{nullptr}, block3{nullptr};
//     torch::nn::Conv2d to_img_4x4{nullptr}, to_img_8x8{nullptr}, to_img_16x16{nullptr}, to_img_28x28{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(ProGANGenerator);
//
// // Discriminator Block
// struct DiscriminatorBlockImpl : torch::nn::Module {
//     DiscriminatorBlockImpl(int in_channels, int out_channels) {
//         conv = register_module("conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, out_channels, 4).stride(2).padding(1)));
//         bn = register_module("bn", torch::nn::BatchNorm2d(out_channels));
//         lrelu = register_module("lrelu", torch::nn::LeakyReLU(
//             torch::nn::LeakyReLUOptions().negative_slope(0.2)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = lrelu->forward(bn->forward(conv->forward(x)));
//         return x;
//     }
//
//     torch::nn::Conv2d conv{nullptr};
//     torch::nn::BatchNorm2d bn{nullptr};
//     torch::nn::LeakyReLU lrelu{nullptr};
// };
// TORCH_MODULE(DiscriminatorBlock);
//
// // Discriminator
// struct ProGANDiscriminatorImpl : torch::nn::Module {
//     ProGANDiscriminatorImpl() {
//         from_img_28x28 = register_module("from_img_28x28", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(1, 32, 3).stride(1).padding(1)));
//         from_img_16x16 = register_module("from_img_16x16", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(1, 64, 1).stride(1)));
//         from_img_8x8 = register_module("from_img_8x8", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(1, 128, 1).stride(1)));
//         from_img_4x4 = register_module("from_img_4x4", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(1, 256, 1).stride(1)));
//         block3 = register_module("block3", DiscriminatorBlock(32, 64)); // 28x28 -> 16x16
//         block2 = register_module("block2", DiscriminatorBlock(64, 128)); // 16x16 -> 8x8
//         block1 = register_module("block1", DiscriminatorBlock(128, 256)); // 8x8 -> 4x4
//         final = register_module("final", torch::nn::Linear(256 * 4 * 4, 1));
//         lrelu = register_module("lrelu", torch::nn::LeakyReLU(
//             torch::nn::LeakyReLUOptions().negative_slope(0.2)));
//     }
//
//     torch::Tensor forward(torch::Tensor x, int resolution_level, float alpha) {
//         auto batch_size = x.size(0);
//
//         if (resolution_level == 0) {
//             x = lrelu->forward(from_img_4x4->forward(x)); // [batch, 256, 4, 4]
//         } else if (resolution_level == 1 || resolution_level == 2) {
//             x = lrelu->forward(from_img_8x8->forward(x)); // [batch, 128, 8, 8]
//             if (resolution_level == 2) {
//                 auto x_prev = lrelu->forward(from_img_4x4->forward(torch::avg_pool2d(x, 2)));
//                 x = (1 - alpha) * x_prev + alpha * x; // Fade-in
//             }
//             x = block1->forward(x); // [batch, 256, 4, 4]
//         } else if (resolution_level == 3 || resolution_level == 4) {
//             x = lrelu->forward(from_img_16x16->forward(x)); // [batch, 64, 16, 16]
//             if (resolution_level == 4) {
//                 auto x_prev = lrelu->forward(from_img_8x8->forward(torch::avg_pool2d(x, 2)));
//                 x = (1 - alpha) * x_prev + alpha * x; // Fade-in
//             }
//             x = block2->forward(x); // [batch, 128, 8, 8]
//             x = block1->forward(x); // [batch, 256, 4, 4]
//         } else if (resolution_level == 5 || resolution_level == 6) {
//             x = lrelu->forward(from_img_28x28->forward(x)); // [batch, 32, 28, 28]
//             if (resolution_level == 6) {
//                 auto x_prev = lrelu->forward(from_img_16x16->forward(torch::avg_pool2d(x, 2)));
//                 x = (1 - alpha) * x_prev + alpha * x; // Fade-in
//             }
//             x = block3->forward(x); // [batch, 64, 16, 16]
//             x = block2->forward(x); // [batch, 128, 8, 8]
//             x = block1->forward(x); // [batch, 256, 4, 4]
//         } else {
//             throw std::runtime_error("Invalid resolution level");
//         }
//
//         x = x.view({batch_size, -1}); // [batch, 256 * 4 * 4]
//         x = final->forward(x); // [batch, 1]
//         return x;
//     }
//
//     torch::nn::Conv2d from_img_28x28{nullptr}, from_img_16x16{nullptr}, from_img_8x8{nullptr}, from_img_4x4{nullptr};
//     DiscriminatorBlock block3{nullptr}, block2{nullptr}, block1{nullptr};
//     torch::nn::Linear final{nullptr};
//     torch::nn::LeakyReLU lrelu{nullptr};
// };
// TORCH_MODULE(ProGANDiscriminator);
//
// // Custom Dataset for Grayscale Images
// struct ImageDataset : torch::data::Dataset<ImageDataset> {
//     ImageDataset(const std::string& img_dir) {
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
//         image.convertTo(image, CV_32F, 2.0 / 255.0, -1.0); // Normalize to [-1, 1]
//         torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols}, torch::kFloat32);
//         return {img_tensor, torch::Tensor()};
//     }
//
//     torch::optional<size_t> size() const override {
//         return image_paths_.size();
//     }
//
//     std::vector<std::string> image_paths_;
// };
//
// int main() {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Hyperparameters
//         const int latent_dim = 100;
//         const int batch_size = 64;
//         const float lr = 0.001;
//         const float beta1 = 0.0; // As per ProGAN paper
//         const int epochs_per_level = 10;
//         const float alpha_increment = 0.01; // Fade-in speed
//
//         // Initialize models
//         ProGANGenerator generator(latent_dim);
//         ProGANDiscriminator discriminator;
//         generator->to(device);
//         discriminator->to(device);
//
//         // Optimizers
//         torch::optim::Adam g_optimizer(generator->parameters(), torch::optim::AdamOptions(lr).betas({beta1, 0.99}));
//         torch::optim::Adam d_optimizer(discriminator->parameters(), torch::optim::AdamOptions(lr).betas({beta1, 0.99}));
//
//         // Loss function
//         auto bce_loss = torch::nn::BCELoss();
//
//         // Load dataset
//         auto dataset = ImageDataset("./data/images")
//             .map(torch::data::transforms::Stack<>());
//         auto data_loader = torch::data::make_data_loader(
//             dataset, torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));
//
//         // Training schedule: resolution levels (0: 4x4, 1: 8x8 stable, 2: 8x8 fade, 3: 16x16 stable, 4: 16x16 fade, 5: 28x28 stable, 6: 28x28 fade)
//         for (int res_level = 0; res_level <= 6; ++res_level) {
//             float alpha = (res_level % 2 == 0) ? 1.0 : 0.0; // Stable or fade-in
//             int target_size = (res_level <= 0) ? 4 : (res_level <= 2) ? 8 : (res_level <= 4) ? 16 : 28;
//
//             std::cout << "Training at resolution level " << res_level << " (" << target_size << "x" << target_size << ")" << std::endl;
//
//             generator->train();
//             discriminator->train();
//
//             for (int epoch = 0; epoch < epochs_per_level; ++epoch) {
//                 float g_loss_avg = 0.0;
//                 float d_loss_avg = 0.0;
//                 int batch_count = 0;
//
//                 for (auto& batch : *data_loader) {
//                     auto real_imgs = batch.data.to(device);
//                     auto batch_size_current = real_imgs.size(0);
//                     real_imgs = torch::avg_pool2d(real_imgs, real_imgs.size(2) / target_size); // Downsample to target resolution
//
//                     // Train Discriminator
//                     d_optimizer.zero_grad();
//                     auto real_scores = discriminator->forward(real_imgs, res_level, alpha);
//                     auto real_label = torch::ones_like(real_scores, device);
//                     auto d_loss_real = bce_loss->forward(torch::sigmoid(real_scores), real_label);
//
//                     auto z = torch::randn({batch_size_current, latent_dim}, device);
//                     auto fake_imgs = generator->forward(z, res_level, alpha);
//                     auto fake_scores = discriminator->forward(fake_imgs.detach(), res_level, alpha);
//                     auto fake_label = torch::zeros_like(fake_scores, device);
//                     auto d_loss_fake = bce_loss->forward(torch::sigmoid(fake_scores), fake_label);
//
//                     auto d_loss = d_loss_real + d_loss_fake;
//                     d_loss.backward();
//                     d_optimizer.step();
//
//                     // Train Generator
//                     g_optimizer.zero_grad();
//                     fake_scores = discriminator->forward(fake_imgs, res_level, alpha);
//                     auto g_loss = bce_loss->forward(torch::sigmoid(fake_scores), real_label);
//                     g_loss.backward();
//                     g_optimizer.step();
//
//                     d_loss_avg += d_loss.item<float>();
//                     g_loss_avg += g_loss.item<float>();
//                     batch_count++;
//
//                     // Update alpha during fade-in phases
//                     if (res_level % 2 == 1) {
//                         alpha = std::min(1.0f, alpha + alpha_increment);
//                     }
//                 }
//
//                 std::cout << "Epoch [" << epoch + 1 << "/" << epochs_per_level << "] "
//                           << "Discriminator Loss: " << d_loss_avg / batch_count
//                           << ", Generator Loss: " << g_loss_avg / batch_count
//                           << ", Alpha: " << alpha << std::endl;
//
//                 // Save generated image
//                 if (epoch == epochs_per_level - 1) {
//                     torch::NoGradGuard no_grad;
//                     auto z = torch::randn({1, latent_dim}, device);
//                     auto gen_img = generator->forward(z, res_level, alpha).squeeze().to(torch::kCPU);
//                     cv::Mat img(target_size, target_size, CV_32F, gen_img.data_ptr<float>());
//                     img = (img + 1.0) * 127.5; // Denormalize [-1, 1] to [0, 255]
//                     img.convertTo(img, CV_8U);
//                     cv::imwrite("generated_progan_level_" + std::to_string(res_level) + "_epoch_" + std::to_string(epoch + 1) + ".jpg", img);
//                 }
//             }
//         }
//
//         // Save models
//         torch::save(generator, "progan_generator.pt");
//         torch::save(discriminator, "progan_discriminator.pt");
//         std::cout << "Models saved as progan_generator.pt and progan_discriminator.pt" << std::endl;
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
    ProGAN::ProGAN(int num_classes, int in_channels)
    {
    }

    ProGAN::ProGAN(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void ProGAN::reset()
    {
    }

    auto ProGAN::forward(std::initializer_list<std::any> tensors) -> std::any
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
