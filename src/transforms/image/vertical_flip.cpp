#include "include/transforms/image/vertical_flip.h"


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