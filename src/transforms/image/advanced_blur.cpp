#include "include/transforms/image/advanced_blur.h"


namespace xt::transforms::image {

    AdvancedBlur::AdvancedBlur() : block_size_(11), c_(2.0), adaptive_method_flag_(cv::ADAPTIVE_THRESH_GAUSSIAN_C) {}

    AdvancedBlur::AdvancedBlur(int block_size, double c, const std::string& adaptive_method)
        : block_size_(block_size), c_(c) {

        if (block_size <= 1 || block_size % 2 == 0) {
            throw std::invalid_argument("AdvancedBlur block_size must be an odd integer greater than 1.");
        }

        if (adaptive_method == "mean") {
            adaptive_method_flag_ = cv::ADAPTIVE_THRESH_MEAN_C;
        } else if (adaptive_method == "gaussian") {
            adaptive_method_flag_ = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
        } else {
            throw std::invalid_argument("Adaptive method must be 'mean' or 'gaussian'.");
        }
    }

    auto AdvancedBlur::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("AdvancedBlur::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to AdaptiveThreshold is not defined.");
        }

        // 2. --- Prepare Image for OpenCV ---
        // Adaptive thresholding requires a single-channel, 8-bit integer image.
        torch::Tensor grayscale_tensor;
        if (input_tensor.size(0) == 3) { // Color image
            auto weights = torch::tensor({0.299, 0.587, 0.114}, input_tensor.options()).view({3, 1, 1});
            grayscale_tensor = (input_tensor * weights).sum(0, true);
        } else if (input_tensor.size(0) == 1) { // Already grayscale
            grayscale_tensor = input_tensor;
        } else {
            throw std::invalid_argument("Input must be a 1-channel or 3-channel image.");
        }

        // Convert the float tensor (range 0-1) to an 8-bit integer mat (range 0-255)
        cv::Mat input_mat_8u = xt::utils::image::tensor_to_mat_8u(grayscale_tensor);


        // 3. --- Apply Adaptive Threshold ---
        cv::Mat binary_mat;
        cv::adaptiveThreshold(
            input_mat_8u,               // Source 8-bit single-channel image
            binary_mat,                 // Destination image
            255,                        // Max value to assign (we'll scale it down later)
            adaptive_method_flag_,      // Adaptive method
            cv::THRESH_BINARY,          // Thresholding type
            block_size_,                // Size of the pixel neighborhood
            c_                          // Constant subtracted from the mean
        );

        // 4. --- Convert back to LibTorch Tensor (with values 0.0 and 1.0) ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_local(binary_mat);
        
        // The output from OpenCV is 0 or 255. We scale it to 0 or 1 to match ML conventions.
        output_tensor = output_tensor / 255.0f;
        
        // Add channel dimension back to get [1, H, W]
        return output_tensor.unsqueeze(0);
    }

} // namespace xt::transforms::image