#include "include/transforms/image/cut_mix.h"

// #include "transforms/image/cutmix.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy batch of data
//     int batch_size = 4;
//     int num_classes = 10;
//
//     // Batch of images
//     torch::Tensor images = torch::rand({batch_size, 3, 32, 32});
//     // Batch of integer labels
//     torch::Tensor labels = torch::randint(0, num_classes, {batch_size}, torch::kLong);
//
//     std::cout << "Original labels: " << labels << std::endl;
//
//     // 2. Instantiate the CutMix transform
//     xt::transforms::image::CutMix cutmix_transform(1.0f, /*p=*/1.0f); // Apply every time for demo
//
//     // 3. Apply the transform to the batch
//     auto result_any = cutmix_transform.forward({images, labels});
//     auto result_pair = std::any_cast<std::pair<torch::Tensor, torch::Tensor>>(result_any);
//
//     torch::Tensor mixed_images = result_pair.first;
//     torch::Tensor mixed_labels = result_pair.second;
//
//     // 4. Check the output
//     std::cout << "\nShape of mixed images: " << mixed_images.sizes() << std::endl;
//     std::cout << "Shape of mixed labels: " << mixed_labels.sizes() << std::endl; // Should be [B, NumClasses]
//     std::cout << "Mixed labels (one-hot format):\n" << mixed_labels << std::endl;
//     // Each row in the mixed_labels tensor will have two non-zero values that sum to 1.0
//
//     // You would then pass these mixed batches to your model:
//     // auto output = model->forward(mixed_images);
//     // auto loss = some_loss_function(output, mixed_labels); // Use with a loss that accepts soft labels
//
//     return 0;
// }


namespace xt::transforms::image {

    // Default constructor
    CutMix::CutMix() : alpha_(1.0f), p_(0.5f) {}

    // Main constructor
    CutMix::CutMix(float alpha, float p) : alpha_(alpha), p_(p) {
        if (alpha_ <= 0) {
            throw std::invalid_argument("CutMix alpha must be positive.");
        }
        if (p_ < 0.0f || p_ > 1.0f) {
            throw std::invalid_argument("CutMix probability must be between 0 and 1.");
        }
    }

    // Private helper to generate the bounding box for the patch
    torch::Tensor CutMix::_generate_bbox(int64_t H, int64_t W, double lambda) {
        // Calculate the patch dimensions from lambda
        auto cut_ratio = std::sqrt(1.0 - lambda);
        auto cut_h = static_cast<int64_t>(H * cut_ratio);
        auto cut_w = static_cast<int64_t>(W * cut_ratio);

        // Randomly choose the center of the patch
        auto cx = torch::randint(0, W, {1}).item<int64_t>();
        auto cy = torch::randint(0, H, {1}).item<int64_t>();

        // Calculate top-left and bottom-right corners
        auto bbx1 = cx - cut_w / 2;
        auto bby1 = cy - cut_h / 2;
        auto bbx2 = cx + cut_w / 2;
        auto bby2 = cy + cut_h / 2;

        // Clamp the coordinates to be within the image boundaries
        bbx1 = std::max((int64_t)0, bbx1);
        bby1 = std::max((int64_t)0, bby1);
        bbx2 = std::min(W, bbx2);
        bby2 = std::min(H, bby2);

        return torch::tensor({bbx1, bby1, bbx2, bby2}, torch::kLong);
    }

    auto CutMix::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Decide whether to apply the transform based on probability p ---
        if (torch::rand({1}).item<float>() > p_) {
            // If not applying, return the original inputs directly.
            // This assumes the inputs are already in a std::pair or can be reconstructed.
            // For a robust API, let's just return the original tensors in a new pair.
            return std::make_pair(
                std::any_cast<torch::Tensor>(tensors.begin()[0]),
                std::any_cast<torch::Tensor>(tensors.begin()[1])
            );
        }

        // --- 1. Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("CutMix::forward expects exactly 2 tensors: a batch of images and a batch of labels.");
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
        int64_t H = images.size(2);
        int64_t W = images.size(3);

        // --- 2. Generate Shuffled Indices and Lambda ---
        // Create a randomly shuffled version of the batch indices
        torch::Tensor rand_indices = torch::randperm(batch_size, images.options().dtype(torch::kLong));

        // Sample lambda from a Beta distribution using your robust utility
        double lambda = xt::utils::random::sample_beta(alpha_, alpha_);

        // --- 3. Generate Bounding Box and Mix Images ---
        torch::Tensor bbox = _generate_bbox(H, W, lambda);
        auto bbx1 = bbox[0].item<int64_t>();
        auto bby1 = bbox[1].item<int64_t>();
        auto bbx2 = bbox[2].item<int64_t>();
        auto bby2 = bbox[3].item<int64_t>();

        // Create a copy of the images to modify
        auto mixed_images = images.clone();

        // Use advanced indexing and slicing to paste the patch from the shuffled batch onto the original
        // This is a powerful feature of LibTorch.
        mixed_images.index_put_(
            {rand_indices, torch::indexing::Slice(), torch::indexing::Slice(bby1, bby2), torch::indexing::Slice(bbx1, bbx2)},
            images.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(bby1, bby2), torch::indexing::Slice(bbx1, bbx2)})
        );

        // --- 4. Adjust Lambda and Mix Labels ---
        // Adjust lambda to match the true area of the patch after clamping
        lambda = 1.0 - (static_cast<double>((bbx2 - bbx1) * (bby2 - bby1)) / (H * W));

        // Determine number of classes for one-hot encoding
        // Find the max label value if labels are not one-hot
        int64_t num_classes;
        bool is_one_hot = (labels.dim() > 1 && labels.size(1) > 1);
        if (is_one_hot) {
            num_classes = labels.size(1);
        } else {
            num_classes = (labels.max().item<int64_t>() + 1);
        }

        // Create one-hot encoded labels for mixing if they aren't already
        torch::Tensor one_hot_labels = is_one_hot ? labels.to(torch::kFloat32) : torch::nn::functional::one_hot(labels, num_classes).to(torch::kFloat32);

        // Get the labels from the shuffled batch
        auto shuffled_labels = one_hot_labels.index_select(0, rand_indices);

        // Mix the labels according to the adjusted lambda
        auto mixed_labels = one_hot_labels * lambda + shuffled_labels * (1.0 - lambda);

        return std::make_pair(mixed_images, mixed_labels);
    }

} // namespace xt::transforms::image