#include <transforms/image/random_flip.h>


// --- Example Main (for testing) ---
// #include "transforms/image/random_flip.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a dummy image with some asymmetry to see the flip clearly.
//     // Let's draw the letter 'F' on it.
//     torch::Tensor image = torch::zeros({3, 200, 200});
//     // Vertical bar
//     image.index_put_({torch::indexing::Slice(), torch::indexing::Slice(40, 160), torch::indexing::Slice(40, 60)}, 1.0);
//     // Top horizontal bar
//     image.index_put_({torch::indexing::Slice(), torch::indexing::Slice(40, 60), torch::indexing::Slice(60, 140)}, 1.0);
//     // Middle horizontal bar
//     image.index_put_({torch::indexing::Slice(), torch::indexing::Slice(90, 110), torch::indexing::Slice(60, 120)}, 1.0);
//
//     cv::imwrite("flip_before.png", xt::utils::image::tensor_to_mat_8u(image));
//     std::cout << "Saved flip_before.png" << std::endl;
//
//     // --- Example 1: Horizontal Flip ---
//     std::cout << "\n--- Applying Horizontal Flip ---" << std::endl;
//     xt::transforms::image::RandomFlip flipper_h(xt::transforms::image::FlipOrientation::Horizontal, 1.0);
//     torch::Tensor flipped_h = std::any_cast<torch::Tensor>(flipper_h.forward({image}));
//     cv::imwrite("flipped_horizontal.png", xt::utils::image::tensor_to_mat_8u(flipped_h));
//     std::cout << "Saved flipped_horizontal.png" << std::endl;
//
//     // --- Example 2: Vertical Flip ---
//     std::cout << "\n--- Applying Vertical Flip ---" << std::endl;
//     xt::transforms::image::RandomFlip flipper_v(xt::transforms::image::FlipOrientation::Vertical, 1.0);
//     torch::Tensor flipped_v = std::any_cast<torch::Tensor>(flipper_v.forward({image}));
//     cv::imwrite("flipped_vertical.png", xt::utils::image::tensor_to_mat_8u(flipped_v));
//     std::cout << "Saved flipped_vertical.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomFlip::RandomFlip() : RandomFlip(FlipOrientation::Horizontal, 0.5) {}

    RandomFlip::RandomFlip(FlipOrientation orientation, double p)
        : orientation_(orientation), p_(p) {

        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }
        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomFlip::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomFlip::forward received an empty list.");
        }
        torch::Tensor img = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!img.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomFlip is not defined.");
        }
        if (img.dim() != 3) {
            throw std::invalid_argument("RandomFlip expects a 3D tensor (C, H, W).");
        }

        // --- Determine Flip Dimension ---
        // For a (C, H, W) tensor:
        // - Horizontal flip is along the width dimension (dim 2).
        // - Vertical flip is along the height dimension (dim 1).
        int dim_to_flip = (orientation_ == FlipOrientation::Horizontal) ? 2 : 1;

        // --- Apply Flip ---
        // torch::flip is highly optimized and returns a view.
        return torch::flip(img, {dim_to_flip});
    }

} // namespace xt::transforms::image