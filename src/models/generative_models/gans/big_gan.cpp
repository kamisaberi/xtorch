#include "include/models/generative_models/gans/big_gan.h"


using namespace std;




//
// #include <torch/torch.h>
// #include <fstream>
// #include <iostream>
// #include <vector>
// #include <string>
// #include <random>
//
// // Conditional Batch Normalization
// struct ConditionalBatchNorm : torch::nn::Module {
//     ConditionalBatchNorm(int64_t num_features, int64_t num_classes)
//         : bn(num_features),
//           gamma_embed(torch::nn::Linear(num_classes, num_features)),
//           beta_embed(torch::nn::Linear(num_classes, num_features)) {
//         register_module("bn", bn);
//         register_module("gamma_embed", gamma_embed);
//         register_module("beta_embed", beta_embed);
//     }
//
//     torch::Tensor forward(torch::Tensor x, torch::Tensor y) {
//         auto out = bn->forward(x);
//         auto gamma = gamma_embed->forward(y).unsqueeze(2).unsqueeze(3);
//         auto beta = beta_embed->forward(y).unsqueeze(2).unsqueeze(3);
//         return gamma * out + beta;
//     }
//
//     torch::nn::BatchNorm2d bn;
//     torch::nn::Linear gamma_embed, beta_embed;
// };
//
// // Spectral Normalization (approximate via power iteration)
// struct SpectralNormConv : torch::nn::Module {
//     SpectralNormConv(torch::nn::Conv2dOptions options)
//         : conv(options) {
//         register_module("conv", conv);
//         u = register_buffer("u", torch::randn({1, options.out_channels()}).normal_(0, 1));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Power iteration for spectral norm
//         auto weight = conv->weight.view({conv->weight.size(0), -1});
//         auto v = torch::matmul(weight.t(), u.t()).t();
//         v = v / (v.norm() + 1e-8);
//         auto sigma = torch::matmul(u, torch::matmul(weight, v)).item<float>();
//         return conv->forward(x) / (sigma + 1e-8);
//     }
//
//     torch::nn::Conv2d conv;
//     torch::Tensor u;
// };
//
// // BigGAN Generator Block
// struct GeneratorBlock : torch::nn::Module {
//     GeneratorBlock(int64_t in_channels, int64_t out_channels, int64_t num_classes)
//         : bn1(in_channels, num_classes),
//           conv1(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)),
//           bn2(out_channels, num_classes),
//           conv2(torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1)) {
//         register_module("bn1", bn1);
//         conv1 = SpectralNormConv(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1));
//         register_module("conv1", conv1);
//         register_module("bn2", bn2);
//         conv2 = SpectralNormConv(torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1));
//         register_module("conv2", conv2);
//     }
//
//     torch::Tensor forward(torch::Tensor x, torch::Tensor y) {
//         auto out = torch::relu(bn1->forward(x, y));
//         out = torch::upsample_nearest2d(out, 2);
//         out = conv1->forward(out);
//         out = torch::relu(bn2->forward(out, y));
//         out = conv2->forward(out);
//         return out;
//     }
//
//     ConditionalBatchNorm bn1, bn2;
//     SpectralNormConv conv1, conv2;
// };
//
// // BigGAN Generator
// struct BigGANGenerator : torch::nn::Module {
//     BigGANGenerator(int64_t z_dim, int64_t num_classes)
//         : linear(torch::nn::Linear(z_dim + num_classes, 4 * 4 * 256)),
//           block1(256, 128, num_classes),
//           block2(128, 64, num_classes),
//           bn(64, num_classes),
//           conv(torch::nn::Conv2dOptions(64, 1, 3).padding(1)) {
//         register_module("linear", linear);
//         register_module("block1", block1);
//         register_module("block2", block2);
//         register_module("bn", bn);
//         conv = SpectralNormConv(torch::nn::Conv2dOptions(64, 1, 3).padding(1));
//         register_module("conv", conv);
//     }
//
//     torch::Tensor forward(torch::Tensor z, torch::Tensor y) {
//         auto input = torch::cat({z, y}, 1);
//         auto out = linear->forward(input).view({-1, 256, 4, 4});
//         out = block1->forward(out, y);
//         out = block2->forward(out, y);
//         out = torch::relu(bn->forward(out, y));
//         out = torch::tanh(conv->forward(out));
//         return out;
//     }
//
//     torch::nn::Linear linear;
//     GeneratorBlock block1, block2;
//     ConditionalBatchNorm bn;
//     SpectralNormConv conv;
// };
//
// // Read MNIST image (optional for training)
// std::vector<float> read_mnist_image(const std::string& path, int idx) {
//     std::ifstream file(path, std::ios::binary);
//     if (!file) {
//         std::cerr << "Failed to open MNIST file: " << path << std::endl;
//         return {};
//     }
//     file.seekg(16 + idx * 28 * 28); // Skip header
//     std::vector<float> img(28 * 28);
//     for (int i = 0; i < 28 * 28; ++i) {
//         unsigned char pixel;
//         file.read((char*)&pixel, 1);
//         img[i] = pixel / 255.0f * 2.0f - 1.0f; // Normalize to [-1, 1]
//     }
//     return img;
// }
//
// int main() {
//     // Set device
//     torch::Device device(torch::kCUDA);
//     if (!torch::cuda::is_available()) {
//         std::cerr << "CUDA not available, using CPU." << std::endl;
//         device = torch::Device(torch::kCPU);
//     }
//
//     // Initialize generator
//     const int64_t z_dim = 128;
//     const int64_t num_classes = 10; // MNIST classes
//     BigGANGenerator generator(z_dim, num_classes);
//     generator->to(device);
//
//     // Generate random noise and class label
//     auto z = torch::randn({1, z_dim}, device);
//     auto y_idx = torch::randint(0, num_classes, {1}, device);
//     auto y = torch::zeros({1, num_classes}, device);
//     y[0][y_idx.item<int64_t>()] = 1.0; // One-hot encoding
//     torch::NoGradGuard no_grad;
//
//     // Forward pass
//     auto output = generator->forward(z, y);
//     std::cout << "Generated image shape: " << output.sizes() << std::endl;
//
//     // Save generated image (64x64 grayscale)
//     std::ofstream out("generated_image.raw", std::ios::binary);
//     auto cpu_output = output.cpu().detach().contiguous();
//     auto data = cpu_output.data_ptr<float>();
//     for (size_t i = 0; i < 64 * 64; ++i) {
//         unsigned char pixel = static_cast<unsigned char>((data[i] + 1.0f) * 127.5f);
//         out.write((char*)&pixel, 1);
//     }
//     out.close();
//     std::cout << "Saved generated image to generated_image.raw (64x64 grayscale)" << std::endl;
//
//     // Optional: Load MNIST for training (example)
//     auto mnist_img = read_mnist_image("train-images-idx3-ubyte", 0);
//     if (!mnist_img.empty()) {
//         std::cout << "Loaded MNIST image, size: " << mnist_img.size() << std::endl;
//     }
//
//     return 0;
// }









namespace xt::models
{
    BigGAN::BigGAN(int num_classes, int in_channels)
    {
    }

    BigGAN::BigGAN(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void BigGAN::reset()
    {
    }

    auto BigGAN::forward(std::initializer_list<std::any> tensors) -> std::any
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
