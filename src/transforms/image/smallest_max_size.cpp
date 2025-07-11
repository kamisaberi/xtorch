#include "include/transforms/image/smallest_max_size.h"



namespace xt::transforms::image {

    SmallestMaxSize::SmallestMaxSize() : max_size_(256), interpolation_flag_(cv::INTER_LINEAR) {}

    SmallestMaxSize::SmallestMaxSize(int max_size, const std::string& interpolation)
        : max_size_(max_size) {

        if (max_size_ <= 0) {
            throw std::invalid_argument("SmallestMaxSize max_size must be a positive integer.");
        }

        if (interpolation == "area") {
            interpolation_flag_ = cv::INTER_AREA;
        } else if (interpolation == "linear") {
            interpolation_flag_ = cv::INTER_LINEAR;
        } else if (interpolation == "cubic") {
            interpolation_flag_ = cv::INTER_CUBIC;
        } else if (interpolation == "nearest") {
            interpolation_flag_ = cv::INTER_NEAREST;
        } else {
            throw std::invalid_argument("Unsupported interpolation method for SmallestMaxSize.");
        }
    }

    auto SmallestMaxSize::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("SmallestMaxSize::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined() || input_tensor.dim() != 3) {
            throw std::invalid_argument("SmallestMaxSize expects a defined 3D image tensor (C, H, W).");
        }

        const int64_t H = input_tensor.size(1);
        const int64_t W = input_tensor.size(2);

        // 2. --- Determine if resizing is needed ---
        int64_t smallest_side = std::min(H, W);

        if (smallest_side == max_size_) {
            // Image's smallest side is already the target size.
            return input_tensor;
        }

        // 3. --- Calculate New Dimensions ---
        double scale = static_cast<double>(max_size_) / smallest_side;

        // Use `lround` for proper rounding to the nearest integer
        int64_t new_H = static_cast<int64_t>(std::lround(H * scale));
        int64_t new_W = static_cast<int64_t>(std::lround(W * scale));

        // --- 4. Perform the Resize ---
        cv::Mat input_mat = xt::utils::image::tensor_to_mat_local(input_tensor);
        cv::Mat resized_mat;

        cv::resize(
            input_mat,
            resized_mat,
            cv::Size(new_W, new_H),
            0, 0, // fx and fy are 0 when dsize is specified
            interpolation_flag_
        );

        // --- 5. Convert back to LibTorch Tensor ---
        return xt::utils::image::mat_to_tensor_local(resized_mat);
    }

} // namespace xt::transforms::image