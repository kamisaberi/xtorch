#include <transforms/image/flip.h>


// #include "transforms/image/flip.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor that is not symmetric
//     //    so we can clearly see the flip.
//     torch::Tensor image = torch::zeros({3, 100, 200});
//     // Add a white rectangle in the top-left corner.
//     image.slice(1, 10, 40).slice(2, 10, 60) = 1.0;
//
//     // --- Example 1: Horizontal Flip ---
//     xt::transforms::image::Flip horizontal_flipper("horizontal");
//     torch::Tensor h_flipped_image = std::any_cast<torch::Tensor>(horizontal_flipper.forward({image}));
//     // In the h_flipped_image, the white rectangle should now be in the top-right corner.
//     // We can check the mean value of that region.
//     float top_right_mean_h = h_flipped_image.slice(1, 10, 40).slice(2, 140, 190).mean().item<float>();
//     std::cout << "Mean of top-right region after horizontal flip: " << top_right_mean_h << " (should be ~1.0)" << std::endl;
//
//
//     // --- Example 2: Vertical Flip ---
//     xt::transforms::image::Flip vertical_flipper("vertical");
//     torch::Tensor v_flipped_image = std::any_cast<torch::Tensor>(vertical_flipper.forward({image}));
//     // In the v_flipped_image, the white rectangle should now be in the bottom-left corner.
//     float bottom_left_mean_v = v_flipped_image.slice(1, 60, 90).slice(2, 10, 60).mean().item<float>();
//     std::cout << "Mean of bottom-left region after vertical flip: " << bottom_left_mean_v << " (should be ~1.0)" << std::endl;
//
//
//     // --- Example 3: Both Flips ---
//     xt::transforms::image::Flip both_flipper("both");
//     torch::Tensor both_flipped_image = std::any_cast<torch::Tensor>(both_flipper.forward({image}));
//     // This is equivalent to rotating the image by 180 degrees.
//     // The white rectangle should be in the bottom-right corner.
//     float bottom_right_mean_both = both_flipped_image.slice(1, 60, 90).slice(2, 140, 190).mean().item<float>();
//     std::cout << "Mean of bottom-right region after both flips: " << bottom_right_mean_both << " (should be ~1.0)" << std::endl;
//
//
//     return 0;
// }

namespace xt::transforms::image {

    // Default to horizontal flip, which is the most common augmentation.
    Flip::Flip() : flip_code_(1) {}

    Flip::Flip(const std::string& mode) {
        if (mode == "horizontal") {
            flip_code_ = 1;
        } else if (mode == "vertical") {
            flip_code_ = 0;
        } else if (mode == "both") {
            flip_code_ = -1;
        } else {
            throw std::invalid_argument("Flip mode must be 'horizontal', 'vertical', or 'both'.");
        }
    }

    auto Flip::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Flip::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to Flip is not defined.");
        }

        // 2. --- Convert to OpenCV Mat ---
        cv::Mat input_mat = xt::utils::image::tensor_to_mat_local(input_tensor);

        // 3. --- Apply the Flip operation ---
        cv::Mat flipped_mat;
        cv::flip(input_mat, flipped_mat, flip_code_);

        // 4. --- Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_local(flipped_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image