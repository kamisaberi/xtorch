#include <transforms/image/vertical_flip.h>



// #include "transforms/image/vertical_flip.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor that is not symmetric
//     //    so we can clearly see the flip.
//     torch::Tensor image = torch::zeros({3, 200, 100});
//     // Add a white rectangle in the top-left corner.
//     image.slice(1, 10, 60).slice(2, 10, 40) = 1.0;
//
//     // 2. Instantiate the transform
//     xt::transforms::image::VerticalFlip flipper;
//
//     // 3. Apply the transform
//     std::any result_any = flipper.forward({image});
//     torch::Tensor flipped_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Original image shape: " << image.sizes() << std::endl;
//     std::cout << "Flipped image shape: " << flipped_image.sizes() << std::endl;
//
//     // To verify, check if the mean value in the new bottom-left region is high.
//     // The original top-left was (10:60) on the height axis (200).
//     // The new bottom-left should be (200-60 : 200-10) -> (140:190).
//     float bottom_left_mean = flipped_image.slice(1, 140, 190).slice(2, 10, 40).mean().item<float>();
//     std::cout << "Mean of bottom-left region after flip: " << bottom_left_mean << " (should be ~1.0)" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    VerticalFlip::VerticalFlip() = default;

    auto VerticalFlip::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("VerticalFlip::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to VerticalFlip is not defined.");
        }

        // 2. --- Convert to OpenCV Mat ---
        cv::Mat input_mat = xt::utils::image::tensor_to_mat_local(input_tensor);

        // 3. --- Apply the Vertical Flip operation ---
        cv::Mat flipped_mat;
        // A flip code of 0 specifies a flip around the x-axis (vertical).
        cv::flip(input_mat, flipped_mat, 0);

        // 4. --- Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_local(flipped_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image