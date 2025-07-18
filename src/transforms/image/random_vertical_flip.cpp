#include <transforms/image/random_vertical_flip.h>


// --- Example Main (for testing) ---
// #include "transforms/image/random_vertical_flip.h"
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
//     cv::imwrite("vflip_before.png", xt::utils::image::tensor_to_mat_8u(image));
//     std::cout << "Saved vflip_before.png" << std::endl;
//
//     // --- Apply the Vertical Flip ---
//     std::cout << "\n--- Applying Vertical Flip ---" << std::endl;
//     // Use p=1.0 to guarantee the transform is applied.
//     xt::transforms::image::RandomVerticalFlip flipper(1.0);
//
//     torch::Tensor flipped_v = std::any_cast<torch::Tensor>(flipper.forward({image}));
//
//     cv::imwrite("vflip_after.png", xt::utils::image::tensor_to_mat_8u(flipped_v));
//     std::cout << "Saved vflip_after.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomVerticalFlip::RandomVerticalFlip() : RandomVerticalFlip(0.5) {}

    RandomVerticalFlip::RandomVerticalFlip(double p) : p_(p) {
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }
        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomVerticalFlip::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomVerticalFlip::forward received an empty list.");
        }
        torch::Tensor img = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!img.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomVerticalFlip is not defined.");
        }
        if (img.dim() != 3) {
            throw std::invalid_argument("RandomVerticalFlip expects a 3D tensor (C, H, W).");
        }

        // --- Apply Flip ---
        // For a (C, H, W) tensor, a vertical flip is along the height dimension (dim 1).
        return torch::flip(img, {1});
    }

} // namespace xt::transforms::image