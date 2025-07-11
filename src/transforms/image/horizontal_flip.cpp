#include "include/transforms/image/horizontal_flip.h"

// #include "transforms/image/horizontal_flip.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor that is not symmetric
//     //    so we can clearly see the flip.
//     torch::Tensor image = torch::zeros({3, 100, 200});
//     // Add a white rectangle in the top-left corner.
//     image.slice(1, 10, 40).slice(2, 10, 60) = 1.0;
//
//     // 2. Instantiate the transform
//     xt::transforms::image::HorizontalFlip flipper;
//
//     // 3. Apply the transform
//     std::any result_any = flipper.forward({image});
//     torch::Tensor flipped_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Original image shape: " << image.sizes() << std::endl;
//     std::cout << "Flipped image shape: " << flipped_image.sizes() << std::endl;
//
//     // To verify, check if the mean value in the new top-right region is high.
//     // The original top-left was (10:60) on the width axis (200).
//     // The new top-right should be (200-60 : 200-10) -> (140:190).
//     float top_right_mean = flipped_image.slice(1, 10, 40).slice(2, 140, 190).mean().item<float>();
//     std::cout << "Mean of top-right region after flip: " << top_right_mean << " (should be ~1.0)" << std::endl;
//
//     return 0;
// }

namespace xt::transforms::image {

    HorizontalFlip::HorizontalFlip() = default;

    auto HorizontalFlip::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("HorizontalFlip::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to HorizontalFlip is not defined.");
        }

        // 2. --- Convert to OpenCV Mat ---
        cv::Mat input_mat = xt::utils::image::tensor_to_mat_local(input_tensor);

        // 3. --- Apply the Horizontal Flip operation ---
        cv::Mat flipped_mat;
        // A flip code of 1 specifies a flip around the y-axis (horizontal).
        cv::flip(input_mat, flipped_mat, 1);

        // 4. --- Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_local(flipped_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image