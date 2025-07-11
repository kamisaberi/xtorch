#include "include/transforms/image/median_blur.h"


namespace xt::transforms::image {

    MedianBlur::MedianBlur() : kernel_size_(3) {}

    MedianBlur::MedianBlur(int kernel_size) : kernel_size_(kernel_size) {
        if (kernel_size_ <= 1 || kernel_size_ % 2 == 0) {
            throw std::invalid_argument("MedianBlur kernel_size must be an odd integer greater than 1.");
        }
    }

    auto MedianBlur::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("MedianBlur::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to MedianBlur is not defined.");
        }

        // 2. --- Convert to OpenCV Mat ---
        // medianBlur requires an 8-bit integer Mat (CV_8U).
        cv::Mat input_mat_8u = xt::utils::image::tensor_to_mat_8u(input_tensor);

        // 3. --- Apply Median Blur ---
        cv::Mat blurred_mat;
        cv::medianBlur(
            input_mat_8u,   // source image
            blurred_mat,    // destination image
            kernel_size_    // aperture linear size; must be odd and greater than 1
        );

        // 4. --- Convert back to LibTorch Tensor ---
        // Convert the 8-bit result back to a float tensor in the [0, 1] range.
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_float(blurred_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image