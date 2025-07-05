#include "include/models/generative_models/gans/wgan.h"


using namespace std;

//WGAN GROK
//
// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // Generator Network
// struct GeneratorImpl : torch::nn::Module {
//     GeneratorImpl(int latent_dim) {
//         fc = register_module("fc", torch::nn::Linear(latent_dim, 256 * 4 * 4));
//         conv1 = register_module("conv1", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(256, 128, 4).stride(2).padding(1)));
//         conv2 = register_module("conv2", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1)));
//         conv3 = register_module("conv3", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(64, 1, 3).stride(1).padding(1)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(256));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));
//         bn3 = register_module("bn3", torch::nn::BatchNorm2d(64));
//         relu = register_module("relu", torch::nn::ReLU());
//     }
//
//     torch::Tensor forward(torch::Tensor z) {
//         auto batch_size = z.size(0);
//         z = relu->forward(fc->forward(z)); // [batch, 256 * 4 * 4]
//         z = z.view({batch_size, 256, 4, 4}); // [batch, 256, 4, 4]
//         z = relu->forward(bn1->forward(conv1->forward(z))); // [batch, 128, 8, 8]
//         z = relu->forward(bn2->forward(conv2->forward(z))); // [batch, 64, 16, 16]
//         z = torch::sigmoid(conv3->forward(bn3->forward(z))); // [batch, 1, 28, 28]
//         return z;
//     }
//
//     torch::nn::Linear fc{nullptr};
//     torch::nn::ConvTranspose2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(Generator);
//
// // Critic Network
// struct CriticImpl : torch::nn::Module {
//     CriticImpl() {
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1)));
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1)));
//         conv3 = register_module("conv3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1)));
//         fc = register_module("fc", torch::nn::Linear(256 * 3 * 3, 1));
//         lrelu = register_module("lrelu", torch::nn::LeakyReLU(
//             torch::nn::LeakyReLUOptions().negative_slope(0.2)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = lrelu->forward(conv1->forward(x)); // [batch, 64, 14, 14]
//         x = lrelu->forward(conv2->forward(x)); // [batch, 128, 7, 7]
//         x = lrelu->forward(conv3->forward(x)); // [batch, 256, 3, 3]
//         auto batch_size = x.size(0);
//         x = x.view({batch_size, -1}); // [batch, 256 * 3 * 3]
//         x = fc->forward(x); // [batch, 1]
//         return x;
//     }
//
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
//     torch::nn::Linear fc{nullptr};
//     torch::nn::LeakyReLU lrelu{nullptr};
// };
// TORCH_MODULE(Critic);
//
// // Weight Clipping for Lipschitz Constraint
// void clip_weights(Critic& critic, float clip_value) {
//     for (auto& param : critic->parameters()) {
//         param.data().clamp_(-clip_value, clip_value);
//     }
// }
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
//         image.convertTo(image, CV_32F, 1.0 / 255.0);
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
//         const int n_critic = 5; // Number of critic iterations per generator iteration
//         const float clip_value = 0.01; // Weight clipping value
//         const float lr = 0.00005; // RMSprop learning rate (recommended for WGAN)
//
//         // Initialize models
//         Generator generator(latent_dim);
//         Critic critic;
//         generator->to(device);
//         critic->to(device);
//
//         // Optimizers (RMSprop as per WGAN paper)
//         torch::optim::RMSprop g_optimizer(generator->parameters(), torch::optim::RMSpropOptions(lr));
//         torch::optim::RMSprop c_optimizer(critic->parameters(), torch::optim::RMSpropOptions(lr));
//
//         // Load dataset
//         auto dataset = ImageDataset("./data/images")
//             .map(torch::data::transforms::Stack<>());
//         auto data_loader = torch::data::make_data_loader(
//             dataset, torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));
//
//         // Training loop
//         generator->train();
//         critic->train();
//         const int num_epochs = 50;
//         for (int epoch = 0; epoch < num_epochs; ++epoch) {
//             float c_loss_avg = 0.0;
//             float g_loss_avg = 0.0;
//             int batch_count = 0;
//
//             for (auto& batch : *data_loader) {
//                 auto real_imgs = batch.data.to(device);
//                 auto batch_size_current = real_imgs.size(0);
//
//                 // Train Critic
//                 for (int i = 0; i < n_critic; ++i) {
//                     c_optimizer.zero_grad();
//
//                     // Real images
//                     auto real_scores = critic->forward(real_imgs);
//                     auto loss_critic_real = -real_scores.mean();
//
//                     // Fake images
//                     auto z = torch::randn({batch_size_current, latent_dim}, device);
//                     auto fake_imgs = generator->forward(z).detach();
//                     auto fake_scores = critic->forward(fake_imgs);
//                     auto loss_critic_fake = fake_scores.mean();
//
//                     // Total critic loss
//                     auto loss_critic = loss_critic_real + loss_critic_fake;
//                     loss_critic.backward();
//                     c_optimizer.step();
//
//                     // Clip weights
//                     clip_weights(critic, clip_value);
//
//                     c_loss_avg += loss_critic.item<float>();
//                 }
//
//                 // Train Generator
//                 g_optimizer.zero_grad();
//                 auto z = torch::randn({batch_size_current, latent_dim}, device);
//                 auto fake_imgs = generator->to(device);
//                 auto fake_scores = critic->forward(fake_imgs);
//                 auto loss_generator = -fake_scores.mean();
//                 loss_generator.backward();
//                 g_optimizer.step();
//
//                 g_loss_avg += loss_generator.item<float>();
//                 batch_count++;
//             }
//
//             std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "] "
//                       << "Critic Loss: " << c_loss_avg / (batch_count * n_critic)
//                       << ", Generator Loss: " << g_loss_avg / batch_count << std::endl;
//
//             // Save generated images every 10 epochs
//             if ((epoch + 1) % 10 == 0) {
//                 torch::NoGradGuard no_grad;
//                 auto z = torch::randn({1, latent_dim}, device);
//                 auto gen_img = generator->forward(z).squeeze().to(torch::kCPU);
//                 cv::Mat img(28, 28, CV_32F, gen_img.data_ptr<float>());
//                 img.convertTo(img, CV_8U, 255.0);
//                 cv::imwrite("generated_wgan_epoch_" + std::to_string(epoch + 1) + ".jpg", img);
//             }
//         }
//
//         // Save models
//         torch::save(generator, "wgan_generator.pt");
//         torch::save(critic, "wgan_critic.pt");
//         std::cout << "Models saved as wgan_generator.pt and wgan_critic.pt" << std::endl;
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


    WGAN::GeneratorImpl::GeneratorImpl(int latent_dim)
    {
        fc = register_module("fc", torch::nn::Linear(latent_dim, 256 * 4 * 4));
        conv1 = register_module("conv1", torch::nn::ConvTranspose2d(
                                    torch::nn::ConvTranspose2dOptions(256, 128, 4).stride(2).padding(1)));
        conv2 = register_module("conv2", torch::nn::ConvTranspose2d(
                                    torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1)));
        conv3 = register_module("conv3", torch::nn::ConvTranspose2d(
                                    torch::nn::ConvTranspose2dOptions(64, 1, 3).stride(1).padding(1)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(256));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(64));
        relu = register_module("relu", torch::nn::ReLU());
    }

    torch::Tensor WGAN::GeneratorImpl::forward(torch::Tensor z)
    {
        auto batch_size = z.size(0);
        z = relu->forward(fc->forward(z)); // [batch, 256 * 4 * 4]
        z = z.view({batch_size, 256, 4, 4}); // [batch, 256, 4, 4]
        z = relu->forward(bn1->forward(conv1->forward(z))); // [batch, 128, 8, 8]
        z = relu->forward(bn2->forward(conv2->forward(z))); // [batch, 64, 16, 16]
        z = torch::sigmoid(conv3->forward(bn3->forward(z))); // [batch, 1, 28, 28]
        return z;
    }





    WGAN::WGAN(int num_classes, int in_channels)
    {
    }

    WGAN::WGAN(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void WGAN::reset()
    {
    }

    auto WGAN::forward(std::initializer_list<std::any> tensors) -> std::any
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
