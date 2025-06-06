#include "include/models/generative_models/autoencoders/sparse_autoencoder.h"


using namespace std;


#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>

// Sparse Autoencoder
struct SparseAEImpl : torch::nn::Module {
    SparseAEImpl(int latent_dim = 128) {
        // Encoder
        enc_conv1 = register_module("enc_conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(1, 16, 3).stride(2).padding(1)));
        enc_conv2 = register_module("enc_conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(16, 32, 3).stride(2).padding(1)));
        enc_conv3 = register_module("enc_conv3", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(32, 64, 3).stride(2).padding(1)));
        enc_fc = register_module("enc_fc", torch::nn::Linear(64 * 4 * 4, latent_dim));

        // Decoder
        dec_fc = register_module("dec_fc", torch::nn::Linear(latent_dim, 64 * 4 * 4));
        dec_conv1 = register_module("dec_conv1", torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(64, 32, 3).stride(2).padding(1).output_padding(1)));
        dec_conv2 = register_module("dec_conv2", torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(32, 16, 3).stride(2).padding(1).output_padding(1)));
        dec_conv3 = register_module("dec_conv3", torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(16, 1, 3).stride(2).padding(1).output_padding(1)));

        relu = register_module("relu", torch::nn::ReLU());
        sigmoid = register_module("sigmoid", torch::nn::Sigmoid());
    }

    // Forward pass: returns reconstructed image and latent representation
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        // Encoder
        x = relu->forward(enc_conv1->forward(x)); // [batch, 16, 14, 14] for 28x28 input
        x = relu->forward(enc_conv2->forward(x)); // [batch, 32, 7, 7]
        x = relu->forward(enc_conv3->forward(x)); // [batch, 64, 4, 4]
        x = x.view({-1, 64 * 4 * 4});
        auto latent = relu->forward(enc_fc->forward(x)); // [batch, latent_dim]

        // Decoder
        x = relu->forward(dec_fc->forward(latent)); // [batch, 64 * 4 * 4]
        x = x.view({-1, 64, 4, 4});
        x = relu->forward(dec_conv1->forward(x)); // [batch, 32, 7, 7]
        x = relu->forward(dec_conv2->forward(x)); // [batch, 16, 14, 14]
        x = sigmoid->forward(dec_conv3->forward(x)); // [batch, 1, 28, 28]

        return {x, latent};
    }

    torch::nn::Conv2d enc_conv1{nullptr}, enc_conv2{nullptr}, enc_conv3{nullptr};
    torch::nn::Linear enc_fc{nullptr};
    torch::nn::Linear dec_fc{nullptr};
    torch::nn::ConvTranspose2d dec_conv1{nullptr}, dec_conv2{nullptr}, dec_conv3{nullptr};
    torch::nn::ReLU relu{nullptr};
    torch::nn::Sigmoid sigmoid{nullptr};
};
TORCH_MODULE(SparseAE);

// Custom Dataset for Grayscale Images
struct ImageDataset : torch::data::Dataset<ImageDataset> {
    ImageDataset(const std::string& img_dir) {
        for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
            if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
                image_paths.push_back(entry.path().string());
            }
        }
    }

    torch::data::Example<> get(size_t index) override {
        cv::Mat image = cv::imread(image_paths[index], cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            throw std::runtime_error("Failed to load image: " + image_paths[index]);
        }
        image.convertTo(image, CV_32F, 1.0 / 255.0);

        torch::Tensor img_tensor = torch::from_blob(
            image.data, {1, image.rows, image.cols}, torch::kFloat32
        ); // [1, H, W]

        return {img_tensor, img_tensor}; // Input and target are the same
    }

    torch::optional<size_t> size() const override {
        return image_paths.size();
    }

    std::vector<std::string> image_paths;
};

// Sparsity Loss: KL-divergence to enforce sparse latent activations
torch::Tensor sparsity_loss(torch::Tensor latent, float sparsity_target = 0.05, float sparsity_weight = 0.2) {
    auto rho_hat = torch::mean(latent, 0); // Average activation per latent unit
    auto rho = torch::full_like(rho_hat, sparsity_target);
    auto kl_div = rho * torch::log(rho / rho_hat) + (1 - rho) * torch::log((1 - rho) / (1 - rho_hat));
    return sparsity_weight * torch::sum(kl_div);
}

int main() {
    try {
        // Set device
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

        // Initialize model and optimizer
        SparseAE model(128); // Latent dimension of 128
        model->to(device);
        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));

        // Load dataset
        auto dataset = ImageDataset("./data/images")
            .map(torch::data::transforms::Stack<>());
        auto data_loader = torch::data::make_data_loader(
            dataset, torch::data::DataLoaderOptions().batch_size(32).workers(2));

        // Training loop
        model->train();
        for (int epoch = 0; epoch < 20; ++epoch) {
            float total_loss = 0.0;
            int batch_count = 0;

            for (auto& batch : *data_loader) {
                auto images = batch.data.to(device);
                auto targets = batch.target.to(device);

                optimizer.zero_grad();
                auto [recon, latent] = model->forward(images);
                auto recon_loss = torch::nn::functional::mse_loss(recon, targets);
                auto sparse_loss = sparsity_loss(latent);
                auto loss = recon_loss + sparse_loss;

                loss.backward();
                optimizer.step();

                total_loss += loss.item<float>();
                batch_count++;
            }

            std::cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss / batch_count << std::endl;
        }

        // Save model
        torch::save(model, "sparse_ae.pt");
        std::cout << "Training complete. Model saved as sparse_ae.pt" << std::endl;

        // Inference example
        model->eval();
        cv::Mat test_image = cv::imread("test_image.jpg", cv::IMREAD_GRAYSCALE);
        if (test_image.empty()) {
            std::cerr << "Error: Could not load test image." << std::endl;
            return -1;
        }
        test_image.convertTo(test_image, CV_32F, 1.0 / 255.0);
        torch::Tensor test_tensor = torch::from_blob(
            test_image.data, {1, 1, test_image.rows, test_image.cols}, torch::kFloat32
        ).to(device);

        auto [recon, latent] = model->forward(test_tensor);
        recon = recon.squeeze().to(torch::kCPU);
        cv::Mat reconstructed(
            test_image.rows, test_image.cols, CV_32F, recon.data_ptr<float>()
        );
        reconstructed.convertTo(reconstructed, CV_8U, 255.0);
        cv::imwrite("reconstructed_sae_image.jpg", reconstructed);
        std::cout << "Reconstructed image saved as reconstructed_sae_image.jpg" << std::endl;

        // Check sparsity of latent representation
        auto latent_mean = torch::mean(latent, 0).to(torch::kCPU);
        float avg_activation = latent_mean.mean().item<float>();
        std::cout << "Average latent activation: " << avg_activation << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}



namespace xt::models
{
    SparseAutoEncoder::SparseAutoEncoder(int num_classes, int in_channels)
    {
    }

    SparseAutoEncoder::SparseAutoEncoder(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void SparseAutoEncoder::reset()
    {
    }

    auto SparseAutoEncoder::forward(std::initializer_list<std::any> tensors) -> std::any
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
