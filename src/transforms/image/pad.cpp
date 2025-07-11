
#include "include/transforms/image/pad.h"



namespace xt::transforms::image {

    Pad::Pad() : top_(0), bottom_(0), left_(0), right_(0), border_type_flag_(cv::BORDER_CONSTANT), fill_value_(0.0f) {}

    Pad::Pad(const std::vector<int>& padding, const std::string& mode, float fill_value)
        : fill_value_(fill_value) {

        // Parse the padding vector into top, bottom, left, right
        if (padding.size() == 1) {
            top_ = bottom_ = left_ = right_ = padding[0];
        } else if (padding.size() == 2) {
            top_ = bottom_ = padding[0];
            left_ = right_ = padding[1];
        } else if (padding.size() == 4) {
            top_ = padding[0];
            bottom_ = padding[1];
            left_ = padding[2];
            right_ = padding[3];
        } else {
            throw std::invalid_argument("Padding vector must have 1, 2, or 4 elements: {p}, {p_tb, p_lr}, or {top, bottom, left, right}.");
        }

        // Parse the string mode into an OpenCV enum
        if (mode == "constant") {
            border_type_flag_ = cv::BORDER_CONSTANT;
        } else if (mode == "reflect") {
            // Note: OpenCV has two reflect modes. BORDER_REFLECT_101 is usually what people mean.
            border_type_flag_ = cv::BORDER_REFLECT_101;
        } else if (mode == "replicate") {
            border_type_flag_ = cv::BORDER_REPLICATE;
        } else {
            throw std::invalid_argument("Unsupported padding mode. Must be 'constant', 'reflect', or 'replicate'.");
        }
    }

    auto Pad::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation and Conversion to Mat ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Pad::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to Pad is not defined.");
        }

        cv::Mat input_mat = xt::utils::image::tensor_to_mat_local(input_tensor);

        // 2. --- Apply Padding using cv::copyMakeBorder ---
        cv::Mat padded_mat;

        // The fill value needs to be converted to a cv::Scalar.
        // If the mat has 3 channels, the scalar should have 3 identical values.
        cv::Scalar cv_fill_value;
        if (input_mat.channels() == 3) {
            cv_fill_value = cv::Scalar(fill_value_, fill_value_, fill_value_);
        } else {
            cv_fill_value = cv::Scalar(fill_value_);
        }

        cv::copyMakeBorder(
            input_mat,
            padded_mat,
            top_,
            bottom_,
            left_,
            right_,
            border_type_flag_,
            cv_fill_value // This is only used for BORDER_CONSTANT
        );

        // 3. --- Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_local(padded_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image