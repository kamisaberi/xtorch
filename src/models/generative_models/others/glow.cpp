#include "include/models/generative_models/others/glow.h"


using namespace std;
//GLOW GROK

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>

// ActNorm Layer (Activation Normalization)
struct ActNormImpl : torch::nn::Module {
    ActNormImpl(int channels) {
        scale = register_parameter("scale", torch::ones({1, channels, 1, 1}));
        bias = register_parameter("bias", torch::zeros({1, channels, 1, 1}));
        initialized = false;
    }

    torch::Tensor forward(torch::Tensor x) {
        if (!initialized) {
            auto mean = x.mean({0, 2, 3}, true);
            auto var = x.var({0, 2, 3}, false, true);
            scale = scale / (torch::sqrt(var + 1e-6));
            bias = bias - mean;
            initialized = true;
        }
        return x * scale + bias;
    }

    torch::Tensor inverse(torch::Tensor z) {
        return (z - bias) / (scale + 1e-6);
    }

    torch::Tensor log_det_jacobian(torch::Tensor x) {
        auto batch_size = x.size(0);
        auto spatial_size = x.size(2) * x.size(3);
        return torch::log(torch::abs(scale)).sum() * spatial_size * batch_size;
    }

    torch::Tensor scale, bias;
    bool initialized;
};
TORCH_MODULE(ActNorm);

// Invertible 1x1 Convolution
struct InvConv1x1Impl : torch::nn::Module {
    InvConv1x1Impl(int channels) {
        weight = register_parameter("weight", torch::randn({channels, channels}));
        // Initialize to approximate identity
        auto s = torch::svd(weight).S;
        weight = weight / (s.max() + 1e-6);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto w = weight.unsqueeze(2).unsqueeze(3); // [c_out, c_in, 1, 1]
        return torch::nn::functional::conv2d(x, w);
    }

    torch::Tensor inverse(torch::Tensor z) {
        auto w_inv = torch::inverse(weight).unsqueeze(2).unsqueeze(3);
        return torch::nn::functional::conv2d(z, w_inv);
    }

    torch::Tensor log_det_jacobian(torch::Tensor x) {
        auto batch_size = x.size(0);
        auto spatial_size = x.size(2) * x.size(3);
        auto log_det = torch::logdet(weight);
        return log_det * spatial_size * batch_size;
    }

    torch::Tensor weight;
};
TORCH_MODULE(InvConv1x1);

// Affine Coupling Layer
struct AffineCouplingImpl : torch::nn::Module {
    AffineCouplingImpl(int in_channels) {
        nn = register_module("nn", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels / 2, 128, 3).padding(1)),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, in_channels, 3).padding(1))
        ));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto x_a = x.slice(1, 0, x.size(1) / 2); // First half
        auto x_b = x.slice(1, x.size(1) / 2, x.size(1)); // Second half
        auto s_t = nn->forward(x_a);
        auto s = torch::tanh(s_t.slice(1, 0, s_t.size(1) / 2)); // Scale
        auto t = s_t.slice(1, s_t.size(1) / 2, s_t.size(1)); // Translation
        auto y_a = x_a;
        auto y_b = x_b * torch::exp(s) + t;
        return torch::cat({y_a, y_b}, 1);
    }

    torch::Tensor inverse(torch::Tensor z) {
        auto z_a = z.slice(1, 0, z.size(1) / 2);
        auto z_b = z.slice(1, z.size(1) / 2, z.size(1));
        auto s_t = nn->forward(z_a);
        auto s = torch::tanh(s_t.slice(1, 0, s_t.size(1) / 2));
        auto t = s_t.slice(1, s_t.size(1) / 2, s_t.size(1));
        auto x_a = z_a;
        auto x_b = (z_b - t) * torch::exp(-s);
        return torch::cat({x_a, x_b}, 1);
    }

    torch::Tensor log_det_jacobian(torch::Tensor x) {
        auto x_a = x.slice(1, 0, x.size(1) / 2);
        auto s_t = nn->forward(x_a);
        auto s = torch::tanh(s_t.slice(1, 0, s_t.size(1) / 2));
        auto batch_size = x.size(0);
        auto spatial_size = x.size(2) * x.size(3);
        return s.sum({1, 2, 3}) * batch_size * spatial_size;
    }

    torch::nn::Sequential nn{nullptr};
};
TORCH_MODULE(AffineCoupling);

// Glow Block
struct GlowBlockImpl : torch::nn::Module {
    GlowBlockImpl(int channels) {
        actnorm = register_module("actnorm", ActNorm(channels));
        invconv = register_module("invconv", InvConv1x1(channels));
        coupling = register_module("coupling", AffineCoupling(channels));
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        auto z = actnorm->forward(x);
        auto log_det1 = actnorm->log_det_jacobian(x);
        z = invconv->forward(z);
        auto log_det2 = invconv->log_det_jacobian(z);
        z = coupling->forward(z);
        auto log_det3 = coupling->log_det_jacobian(z);
        auto log_det = log_det1 + log_det2 + log_det3;
        return {z, log_det};
    }

    torch::Tensor inverse(torch::Tensor z) {
        auto x = coupling->inverse(z);
        x = invconv->inverse(x);
        x = actnorm->inverse(x);
        return x;
    }

    ActNorm actnorm{nullptr};
    InvConv1x1 invconv{nullptr};
    AffineCoupling coupling{nullptr};
};
TORCH_MODULE(GlowBlock);

// Glow Model
struct GlowImpl : torch::nn::Module {
    GlowImpl(int channels = 4) {
        block = register_module("block", GlowBlock(channels));
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        auto [z, log_det] = block->forward(x);
        auto log_pz = -0.5 * (z * z + std::log(2 * M_PI)).sum({1, 2, 3}); // Standard normal log-likelihood
        auto log_px = log_pz + log_det; // Log-likelihood of data
        return {z, -log_px.mean()}; // Negative log-likelihood for loss
    }

    torch::Tensor sample(torch::Device device) {
        torch::NoGradGuard no_grad;
        auto z = torch::randn({1, 4, 28, 28}, device); // Sample from standard normal
        auto x = block->inverse(z);
        return torch::sigmoid(x); // Map to [0, 1]
    }

    GlowBlock block{nullptr};
};
TORCH_MODULE(Glow);

// Custom Dataset for Grayscale Images
struct ImageDataset : torch::data::Dataset<ImageDataset> {
    ImageDataset(const std::string& img_dir) {
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
        // Add small noise to prevent numerical issues
        cv::randn(image, 0.0, 1e-4);
        image = cv::max(cv::min(image, 1.0 - 1e-4), 1e-4);
        torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols}, torch::kFloat32);
        // Stack to 4 channels for simplicity
        img_tensor = img_tensor.repeat({1, 4, 1, 1});
        return {img_tensor, torch::Tensor()};
    }

    torch::optional<size_t> size() const override {
        return image_paths_.size();
    }

    std::vector<std::string> image_paths_;
};

int main() {
    try {
        // Set device
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

        // Initialize model
        Glow glow(4);
        glow->to(device);

        // Optimizer
        torch::optim::Adam optimizer(glow->parameters(), torch::optim::AdamOptions(0.001));

        // Load dataset
        auto dataset = ImageDataset("./data/images")
            .map(torch::data::transforms::Stack<>());
        auto data_loader = torch::data::make_data_loader(
            dataset, torch::data::DataLoaderOptions().batch_size(32).workers(2));

        // Training loop
        glow->train();
        for (int epoch = 0; epoch < 20; ++epoch) {
            float total_loss = 0.0;
            int batch_count = 0;

            for (auto& batch : *data_loader) {
                auto images = batch.data.to(device);
                optimizer.zero_grad();
                auto [z, loss] = glow->forward(images);
                loss.backward();
                optimizer.step();
                total_loss += loss.item<float>();
                batch_count++;
            }

            std::cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss / batch_count << std::endl;
        }

        // Save model
        torch::save(glow, "glow.pt");
        std::cout << "Model saved as glow.pt" << std::endl;

        // Inference example
        glow->eval();
        auto generated = glow->sample(device);
        generated = generated.squeeze().slice(0, 0, 1).to(torch::kCPU); // Take first channel
        cv::Mat output(28, 28, CV_32F, generated.data_ptr<float>());
        output.convertTo(output, CV_8U, 255.0);
        cv::imwrite("generated_glow_image.jpg", output);
        std::cout << "Generated image saved as generated_glow_image.jpg" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}






namespace xt::models
{
    Glow::Glow(int num_classes, int in_channels)
    {
    }

    Glow::Glow(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void Glow::reset()
    {
    }

    auto Glow::forward(std::initializer_list<std::any> tensors) -> std::any
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
