#include "include/models/generative_models/gans/pix2pix.h"


using namespace std;

//PIX2PIX GROK

// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // U-Net Generator
// struct UNetGeneratorImpl : torch::nn::Module {
//     UNetGeneratorImpl() {
//         // Encoder
//         enc1 = register_module("enc1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1))); // [batch, 64, 14, 14]
//         enc2 = register_module("enc2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1))); // [batch, 128, 7, 7]
//         enc3 = register_module("enc3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1))); // [batch, 256, 3, 3]
//
//         // Decoder
//         dec3 = register_module("dec3", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(256, 128, 4).stride(2).padding(1))); // [batch, 128, 7, 7]
//         dec2 = register_module("dec2", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(256, 64, 4).stride(2).padding(1))); // [batch, 64, 14, 14]
//         dec1 = register_module("dec1", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(128, 1, 4).stride(2).padding(1))); // [batch, 1, 28, 28]
//
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));
//         bn3 = register_module("bn3", torch::nn::BatchNorm2d(256));
//         bn4 = register_module("bn4", torch::nn::BatchNorm2d(128));
//         bn5 = register_module("bn5", torch::nn::BatchNorm2d(64));
//         relu = register_module("relu", torch::nn::ReLU());
//         tanh = register_module("tanh", torch::nn::Tanh());
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Encoder
//         auto e1 = relu->forward(bn1->forward(enc1->forward(x))); // [batch, 64, 14, 14]
//         auto e2 = relu->forward(bn2->forward(enc2->forward(e1))); // [batch, 128, 7, 7]
//         auto e3 = relu->forward(bn3->forward(enc3->forward(e2))); // [batch, 256, 3, 3]
//
//         // Decoder with skip connections
//         auto d3 = relu->forward(bn4->forward(dec3->forward(e3))); // [batch, 128, 7, 7]
//         auto d2 = relu->forward(bn5->forward(dec2->forward(torch::cat({d3, e2}, 1)))); // [batch, 64, 14, 14]
//         auto d1 = tanh->forward(dec1->forward(torch::cat({d2, e1}, 1))); // [batch, 1, 28, 28]
//
//         return d1;
//     }
//
//     torch::nn::Conv2d enc1{nullptr}, enc2{nullptr}, enc3{nullptr};
//     torch::nn::ConvTranspose2d dec3{nullptr}, dec2{nullptr}, dec1{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr}, bn5{nullptr};
//     torch::nn::ReLU relu{nullptr};
//     torch::nn::Tanh tanh{nullptr};
// };
// TORCH_MODULE(UNetGenerator);
//
// // PatchGAN Discriminator
// struct PatchGANDiscriminatorImpl : torch::nn::Module {
//     PatchGANDiscriminatorImpl() {
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(2, 64, 4).stride(2).padding(1))); // [batch, 64, 14, 14]
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1))); // [batch, 128, 7, 7]
//         conv3 = register_module("conv3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1))); // [batch, 256, 3, 3]
//         conv4 = register_module("conv4", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 1, 3).stride(1).padding(1))); // [batch, 1, 3, 3]
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));
//         bn3 = register_module("bn3", torch::nn::BatchNorm2d(256));
//         lrelu = register_module("lrelu", torch::nn::LeakyReLU(
//             torch::nn::LeakyReLUOptions().negative_slope(0.2)));
//     }
//
//     torch::Tensor forward(torch::Tensor input, torch::Tensor target) {
//         auto x = torch::cat({input, target}, 1); // [batch, 2, 28, 28]
//         x = lrelu->forward(bn1->forward(conv1->forward(x))); // [batch, 64, 14, 14]
//         x = lrelu->forward(bn2->forward(conv2->forward(x))); // [batch, 128, 7, 7]
//         x = lrelu->forward(bn3->forward(conv3->forward(x))); // [batch, 256, 3, 3]
//         x = torch::sigmoid(conv4->forward(x)); // [batch, 1, 3, 3]
//         return x;
//     }
//
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
//     torch::nn::LeakyReLU lrelu{nullptr};
// };
// TORCH_MODULE(PatchGANDiscriminator);
//
// // Paired Image Dataset
// struct PairedImageDataset : torch::data::Dataset<PairedImageDataset> {
//     PairedImageDataset(const std::string& input_dir, const std::string& target_dir) {
//         for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {
//             if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
//                 input_paths_.push_back(entry.path().string());
//                 std::string target_path = target_dir + "/" + entry.path().filename().string();
//                 target_paths_.push_back(target_path);
//             }
//         }
//     }
//
//     torch::data::Example<> get(size_t index) override {
//         // Load input image
//         cv::Mat input_img = cv::imread(input_paths_[index % input_paths_.size()], cv::IMREAD_GRAYSCALE);
//         if (input_img.empty()) {
//             throw std::runtime_error("Failed to load input image: " + input_paths_[index % input_paths_.size()]);
//         }
//         input_img.convertTo(input_img, CV_32F, 2.0 / 255.0, -1.0); // Normalize to [-1, 1]
//         torch::Tensor input_tensor = torch::from_blob(input_img.data, {1, input_img.rows, input_img.cols}, torch::kFloat32);
//
//         // Load target image
//         cv::Mat target_img = cv::imread(target_paths_[index % target_paths_.size()], cv::IMREAD_GRAYSCALE);
//         if (target_img.empty()) {
//             throw std::runtime_error("Failed to load target image: " + target_paths_[index % target_paths_.size()]);
//         }
//         target_img.convertTo(target_img, CV_32F, 2.0 / 255.0, -1.0); // Normalize to [-1, 1]
//         torch::Tensor target_tensor = torch::from_blob(target_img.data, {1, target_img.rows, target_img.cols}, torch::kFloat32);
//
//         return {input_tensor, target_tensor};
//     }
//
//     torch::optional<size_t> size() const override {
//         return input_paths_.size();
//     }
//
//     std::vector<std::string> input_paths_, target_paths_;
// };
//
// int main() {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Hyperparameters
//         const int batch_size = 32;
//         const float lr = 0.0002;
//         const float beta1 = 0.5;
//         const float lambda_l1 = 100.0; // L1 loss weight
//
//         // Initialize models
//         UNetGenerator generator;
//         PatchGANDiscriminator discriminator;
//         generator->to(device);
//         discriminator->to(device);
//
//         // Optimizers
//         torch::optim::Adam g_optimizer(generator->parameters(), torch::optim::AdamOptions(lr).betas({beta1, 0.999}));
//         torch::optim::Adam d_optimizer(discriminator->parameters(), torch::optim::AdamOptions(lr).betas({beta1, 0.999}));
//
//         // Loss functions
//         auto bce_loss = torch::nn::BCELoss();
//         auto l1_loss = torch::nn::L1Loss();
//
//         // Load dataset
//         auto dataset = PairedImageDataset("./data/input_images", "./data/target_images")
//             .map(torch::data::transforms::Stack<>());
//         auto data_loader = torch::data::make_data_loader(
//             dataset, torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));
//
//         // Training loop
//         generator->train();
//         discriminator->train();
//         const int num_epochs = 100;
//         for (int epoch = 0; epoch < num_epochs; ++epoch) {
//             float g_loss_avg = 0.0;
//             float d_loss_avg = 0.0;
//             int batch_count = 0;
//
//             for (auto& batch : *data_loader) {
//                 auto input_imgs = batch.data.to(device);
//                 auto target_imgs = batch.target.to(device);
//                 auto batch_size_current = input_imgs.size(0);
//
//                 // Train Discriminator
//                 d_optimizer.zero_grad();
//
//                 // Real pairs
//                 auto real_output = discriminator->forward(input_imgs, target_imgs);
//                 auto real_label = torch::ones_like(real_output, device);
//                 auto d_loss_real = bce_loss->forward(real_output, real_label);
//
//                 // Fake pairs
//                 auto fake_imgs = generator->forward(input_imgs);
//                 auto fake_output = discriminator->forward(input_imgs, fake_imgs.detach());
//                 auto fake_label = torch::zeros_like(fake_output, device);
//                 auto d_loss_fake = bce_loss->forward(fake_output, fake_label);
//
//                 // Total discriminator loss
//                 auto d_loss = (d_loss_real + d_loss_fake) * 0.5;
//                 d_loss.backward();
//                 d_optimizer.step();
//
//                 // Train Generator
//                 g_optimizer.zero_grad();
//                 auto fake_output_g = discriminator->forward(input_imgs, fake_imgs);
//                 auto g_loss_gan = bce_loss->forward(fake_output_g, real_label);
//                 auto g_loss_l1 = lambda_l1 * l1_loss->forward(fake_imgs, target_imgs);
//                 auto g_loss = g_loss_gan + g_loss_l1;
//                 g_loss.backward();
//                 g_optimizer.step();
//
//                 d_loss_avg += d_loss.item<float>();
//                 g_loss_avg += g_loss.item<float>();
//                 batch_count++;
//             }
//
//             std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "] "
//                       << "Discriminator Loss: " << d_loss_avg / batch_count
//                       << ", Generator Loss: " << g_loss_avg / batch_count << std::endl;
//
//             // Save generated images every 20 epochs
//             if ((epoch + 1) % 20 == 0) {
//                 torch::NoGradGuard no_grad;
//                 auto sample_batch = *data_loader->begin();
//                 auto input_img = sample_batch.data[0].unsqueeze(0).to(device);
//                 auto gen_img = generator->forward(input_img).squeeze().to(torch::kCPU);
//                 cv::Mat img(28, 28, CV_32F, gen_img.data_ptr<float>());
//                 img = (img + 1.0) * 127.5; // Denormalize [-1, 1] to [0, 255]
//                 img.convertTo(img, CV_8U);
//                 cv::imwrite("generated_pix2pix_epoch_" + std::to_string(epoch + 1) + ".jpg", img);
//             }
//         }
//
//         // Save models
//         torch::save(generator, "pix2pix_generator.pt");
//         torch::save(discriminator, "pix2pix_discriminator.pt");
//         std::cout << "Models saved as pix2pix_generator.pt and pix2pix_discriminator.pt" << std::endl;
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


    Pix2Pix::UNetGeneratorImpl::UNetGeneratorImpl()
    {
        // Encoder
        enc1 = register_module("enc1", torch::nn::Conv2d(
                                   torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1)));
        // [batch, 64, 14, 14]
        enc2 = register_module("enc2", torch::nn::Conv2d(
                                   torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1)));
        // [batch, 128, 7, 7]
        enc3 = register_module("enc3", torch::nn::Conv2d(
                                   torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1)));
        // [batch, 256, 3, 3]

        // Decoder
        dec3 = register_module("dec3", torch::nn::ConvTranspose2d(
                                   torch::nn::ConvTranspose2dOptions(256, 128, 4).stride(2).padding(1)));
        // [batch, 128, 7, 7]
        dec2 = register_module("dec2", torch::nn::ConvTranspose2d(
                                   torch::nn::ConvTranspose2dOptions(256, 64, 4).stride(2).padding(1)));
        // [batch, 64, 14, 14]
        dec1 = register_module("dec1", torch::nn::ConvTranspose2d(
                                   torch::nn::ConvTranspose2dOptions(128, 1, 4).stride(2).padding(1)));
        // [batch, 1, 28, 28]

        bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(256));
        bn4 = register_module("bn4", torch::nn::BatchNorm2d(128));
        bn5 = register_module("bn5", torch::nn::BatchNorm2d(64));
        relu = register_module("relu", torch::nn::ReLU());
        tanh = register_module("tanh", torch::nn::Tanh());
    }

    torch::Tensor Pix2Pix::UNetGeneratorImpl::forward(torch::Tensor x)
    {
        // Encoder
        auto e1 = relu->forward(bn1->forward(enc1->forward(x))); // [batch, 64, 14, 14]
        auto e2 = relu->forward(bn2->forward(enc2->forward(e1))); // [batch, 128, 7, 7]
        auto e3 = relu->forward(bn3->forward(enc3->forward(e2))); // [batch, 256, 3, 3]

        // Decoder with skip connections
        auto d3 = relu->forward(bn4->forward(dec3->forward(e3))); // [batch, 128, 7, 7]
        auto d2 = relu->forward(bn5->forward(dec2->forward(torch::cat({d3, e2}, 1)))); // [batch, 64, 14, 14]
        auto d1 = tanh->forward(dec1->forward(torch::cat({d2, e1}, 1))); // [batch, 1, 28, 28]

        return d1;
    }



    Pix2Pix::PatchGANDiscriminatorImpl::PatchGANDiscriminatorImpl()
    {
        conv1 = register_module("conv1", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(2, 64, 4).stride(2).padding(1)));
        // [batch, 64, 14, 14]
        conv2 = register_module("conv2", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1)));
        // [batch, 128, 7, 7]
        conv3 = register_module("conv3", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1)));
        // [batch, 256, 3, 3]
        conv4 = register_module("conv4", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(256, 1, 3).stride(1).padding(1)));
        // [batch, 1, 3, 3]
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(256));
        lrelu = register_module("lrelu", torch::nn::LeakyReLU(
                                    torch::nn::LeakyReLUOptions().negative_slope(0.2)));
    }

    torch::Tensor Pix2Pix::PatchGANDiscriminatorImpl::forward(torch::Tensor input, torch::Tensor target)
    {
        auto x = torch::cat({input, target}, 1); // [batch, 2, 28, 28]
        x = lrelu->forward(bn1->forward(conv1->forward(x))); // [batch, 64, 14, 14]
        x = lrelu->forward(bn2->forward(conv2->forward(x))); // [batch, 128, 7, 7]
        x = lrelu->forward(bn3->forward(conv3->forward(x))); // [batch, 256, 3, 3]
        x = torch::sigmoid(conv4->forward(x)); // [batch, 1, 3, 3]
        return x;
    }





    Pix2Pix::Pix2Pix(int num_classes, int in_channels)
    {
    }

    Pix2Pix::Pix2Pix(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void Pix2Pix::reset()
    {
    }

    auto Pix2Pix::forward(std::initializer_list<std::any> tensors) -> std::any
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
