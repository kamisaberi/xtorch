#include "include/transforms/image/mix_up.h"



// #include "transforms/image/mixup.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy batch of data
//     int batch_size = 4;
//     int num_classes = 10;
//
//     // Create visually distinct images for the demo
//     torch::Tensor image_a = torch::ones({1, 3, 32, 32}); // White
//     torch::Tensor image_b = torch::zeros({1, 3, 32, 32}); // Black
//     torch::Tensor images = torch::cat({image_a, image_b, image_a, image_b}, 0);
//
//     torch::Tensor labels = torch::tensor({0, 1, 0, 1}, torch::kLong);
//
//     std::cout << "Original labels: " << labels << std::endl;
//     std::cout << "Mean of first image (before mixup): " << images[0].mean().item<float>() << std::endl;
//
//     // 2. Instantiate the Mixup transform
//     xt::transforms::image::Mixup mixup_transform(0.4f, /*p=*/1.0f); // Apply every time for demo
//
//     // 3. Apply the transform to the batch
//     auto result_any = mixup_transform.forward({images, labels});
//     auto result_pair = std::any_cast<std::pair<torch::Tensor, torch::Tensor>>(result_any);
//
//     torch::Tensor mixed_images = result_pair.first;
//     torch::Tensor mixed_labels = result_pair.second;
//
//     // 4. Check the output
//     std::cout << "\nShape of mixed images: " << mixed_images.sizes() << std::endl;
//     std::cout << "Shape of mixed labels: " << mixed_labels.sizes() << std::endl; // Should be [B, NumClasses]
//     std::cout << "Mean of first image (after mixup): " << mixed_images[0].mean().item<float>() << std::endl;
//     std::cout << "Mixed labels (soft labels):\n" << mixed_labels << std::endl;
//     // Each row in the mixed_labels tensor will have two non-zero values that sum to 1.0
//     // The mean of the first image will no longer be 1.0, but some value in between,
//     // as it has been mixed with another image from the batch.
//
//     return 0;
// }

namespace xt::transforms::image {

    Mixup::Mixup() : alpha_(0.4f), p_(0.5f) {}

    Mixup::Mixup(float alpha, float p) : alpha_(alpha), p_(p) {
        if (alpha_ <= 0) {
            throw std::invalid_argument("Mixup alpha must be positive.");
        }
        if (p_ < 0.0f || p_ > 1.0f) {
            throw std::invalid_argument("Mixup probability must be between 0 and 1.");
        }
    }

    auto Mixup::forward(std::initializer_list<std::any> tensors) -> std::any {
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
            throw std::invalid_argument("Mixup::forward expects exactly 2 tensors: a batch of images and a batch of labels.");
        }

        torch::Tensor images = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor labels = std::any_cast<torch::Tensor>(any_vec[1]);

        if (!images.defined() || !labels.defined()) {
            throw std::invalid_argument("Input images or labels are not defined.");
        }
        if (images.dim() != 4) {
            throw std::invalid_argument("Input images must be a 4D batch tensor [B, C, H, W].");
        }

        int64_t batch_size = images.size(0);

        // --- 3. Generate Shuffled Indices and Lambda ---
        torch::Tensor rand_indices = torch::randperm(batch_size, images.options().dtype(torch::kLong));

        // Sample lambda from a Beta distribution using your robust utility
        double lambda = xt::utils::random::sample_beta(alpha_, alpha_);

        // --- 4. Mix Images ---
        // Get the images from the shuffled batch
        auto shuffled_images = images.index_select(0, rand_indices);

        // Perform linear interpolation (lerp)
        auto mixed_images = images * lambda + shuffled_images * (1.0 - lambda);

        // --- 5. Mix Labels ---
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

        return std::make_pair(mixed_images, mixed_labels);
    }

} // namespace xt::transforms::image