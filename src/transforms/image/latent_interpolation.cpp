#include <transforms/image/latent_interpolation.h>



//
// #include "transforms/general/latent_interpolation.h"
// #include <iostream>
//
// // --- Dummy Encoder for Demonstration ---
// struct Encoder : torch::nn::Module {
//     torch::nn::Linear layer{nullptr};
//     Encoder(int latent_dim) {
//         layer = register_module("layer", torch::nn::Linear(3 * 32 * 32, latent_dim));
//     }
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::flatten(x, 1);
//         return layer->forward(x);
//     }
// };
//
// int main() {
//     // --- 1. Setup ---
//     int batch_size = 4;
//     int num_classes = 10;
//     int latent_dim = 128;
//
//     // Create a dummy batch of images and labels
//     torch::Tensor images = torch::rand({batch_size, 3, 32, 32});
//     torch::Tensor labels = torch::randint(0, num_classes, {batch_size}, torch::kLong);
//
//     std::cout << "Original labels: " << labels << std::endl;
//
//     // --- 2. Get Latent Vectors ---
//     // In a real scenario, this would be your trained encoder model
//     Encoder encoder(latent_dim);
//     torch::Tensor latents = encoder->forward(images);
//     std::cout << "Shape of latent vectors: " << latents.sizes() << std::endl;
//
//     // --- 3. Instantiate and Apply the LatentInterpolation transform ---
//     xt::transforms::general::LatentInterpolation interpolator(1.0f, /*p=*/1.0f); // Apply every time
//
//     // Note: We pass the LATENTS and LABELS, not the images
//     auto result_any = interpolator.forward({latents, labels});
//     auto result_pair = std::any_cast<std::pair<torch::Tensor, torch::Tensor>>(result_any);
//
//     torch::Tensor mixed_latents = result_pair.first;
//     torch::Tensor mixed_labels = result_pair.second;
//
//     // --- 4. Check the output ---
//     std::cout << "\nShape of mixed latents: " << mixed_latents.sizes() << std::endl;
//     std::cout << "Shape of mixed labels: " << mixed_labels.sizes() << std::endl;
//     std::cout << "Mixed labels (soft labels):\n" << mixed_labels << std::endl;
//     // Each row in mixed_labels will be a soft label, with values summing to 1.0
//
//     // You would then pass `mixed_latents` to the next part of your model (e.g., a classifier)
//     // and use `mixed_labels` with a loss function like CrossEntropyLoss that can handle soft labels.
//
//     return 0;
// }

namespace xt::transforms::general {

    LatentInterpolation::LatentInterpolation() : alpha_(1.0f), p_(0.5f) {}

    LatentInterpolation::LatentInterpolation(float alpha, float p) : alpha_(alpha), p_(p) {
        if (alpha_ <= 0) {
            throw std::invalid_argument("LatentInterpolation alpha must be positive.");
        }
        if (p_ < 0.0f || p_ > 1.0f) {
            throw std::invalid_argument("LatentInterpolation probability must be between 0 and 1.");
        }
    }

    auto LatentInterpolation::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Decide whether to apply the transform ---
        if (torch::rand({1}).item<float>() > p_) {
            return std::make_pair(
                std::any_cast<torch::Tensor>(tensors.begin()[0]),
                std::any_cast<torch::Tensor>(tensors.begin()[1])
            );
        }

        // --- 2. Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("LatentInterpolation::forward expects exactly 2 tensors: latents and labels.");
        }

        torch::Tensor latents = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor labels = std::any_cast<torch::Tensor>(any_vec[1]);

        if (!latents.defined() || !labels.defined()) {
            throw std::invalid_argument("Input latents or labels are not defined.");
        }
        if (latents.dim() != 2) {
            throw std::invalid_argument("Input latents must be a 2D batch tensor [B, LatentDim].");
        }

        int64_t batch_size = latents.size(0);

        // --- 3. Generate Shuffled Indices and Lambda ---
        torch::Tensor rand_indices = torch::randperm(batch_size, latents.options().dtype(torch::kLong));

        // Sample lambda (the interpolation factor) from a Beta distribution
        double lambda = xt::utils::random::sample_beta(alpha_, alpha_);

        // --- 4. Interpolate Latent Vectors ---
        // Get the latent vectors from the shuffled batch
        auto shuffled_latents = latents.index_select(0, rand_indices);

        // Perform linear interpolation (lerp)
        auto mixed_latents = latents * lambda + shuffled_latents * (1.0 - lambda);

        // --- 5. Interpolate Labels ---
        // First, ensure labels are one-hot encoded and float type for mixing
        int64_t num_classes;
        bool is_one_hot = (labels.dim() > 1 && labels.size(1) > 1);
        if (is_one_hot) {
            num_classes = labels.size(1);
        } else {
            num_classes = (labels.max().item<int64_t>() + 1);
        }
        torch::Tensor one_hot_labels = is_one_hot ? labels.to(torch::kFloat32) : torch::nn::functional::one_hot(labels, num_classes).to(torch::kFloat32);

        // Get the labels from the shuffled batch
        auto shuffled_labels = one_hot_labels.index_select(0, rand_indices);

        // Mix the labels with the same lambda
        auto mixed_labels = one_hot_labels * lambda + shuffled_labels * (1.0 - lambda);

        return std::make_pair(mixed_latents, mixed_labels);
    }

} // namespace xt::transforms::general