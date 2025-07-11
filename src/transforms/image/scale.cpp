#include "include/transforms/image/scale.h"
//
//
//
// #include "transforms/image/scale.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor of size [3, 100, 200]
//     torch::Tensor image = torch::rand({3, 100, 200});
//
//     std::cout << "Original image shape: " << image.sizes() << std::endl;
//
//     // --- Example 1: Upscaling ---
//     // Scale the image to 150% of its original size.
//     xt::transforms::image::Scale upscaler(1.5, "cubic"); // Cubic is good for upscaling
//
//     torch::Tensor upscaled_image = std::any_cast<torch::Tensor>(upscaler.forward({image}));
//
//     std::cout << "\n--- Upscaled Image ---" << std::endl;
//     std::cout << "Upscaled image shape: " << upscaled_image.sizes() << std::endl;
//     // New H = 100 * 1.5 = 150. New W = 200 * 1.5 = 300.
//     // Expected output: [3, 150, 300]
//
//
//     // --- Example 2: Downscaling ---
//     // Scale the image to 25% of its original size.
//     xt::transforms::image::Scale downscaler(0.25, "area"); // Area is best for downscaling
//
//     torch::Tensor downscaled_image = std::any_cast<torch::Tensor>(downscaler.forward({image}));
//
//     std::cout << "\n--- Downscaled Image ---" << std::endl;
//     std::cout << "Downscaled image shape: " << downscaled_image.sizes() << std::endl;
//     // New H = 100 * 0.25 = 25. New W = 200 * 0.25 = 50.
//     // Expected output: [3, 25, 50]
//
//     return 0;
// }

namespace xt::transforms::image {

    Scale::Scale() : scale_factor_(1.0), interpolation_flag_(cv::INTER_LINEAR) {}

    Scale::Scale(double scale_factor, const std::string& interpolation)
        : scale_factor_(scale_factor) {

        if (scale_factor <= 0.0) {
            throw std::invalid_argument("Scale factor must be positive.");
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
            throw std::invalid_argument("Unsupported interpolation method for Scale.");
        }
    }

    auto Scale::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Scale::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to Scale is not defined.");
        }

        // If scale factor is 1.0, no change is needed.
        if (scale_factor_ == 1.0) {
            return input_tensor;
        }

        // 2. --- Convert to OpenCV Mat ---
        cv::Mat input_mat = xt::utils::image::tensor_to_mat_local(input_tensor);

        // 3. --- Apply Scaling using cv::resize ---
        cv::Mat scaled_mat;

        // When fx and fy are specified, dsize should be (0,0) for cv::resize
        // to use the scale factors instead of an absolute size.
        cv::resize(
            input_mat,          // source image
            scaled_mat,         // destination image
            cv::Size(),         // target size (dsize=0 to use scale factors)
            scale_factor_,      // fx: scale factor along horizontal axis
            scale_factor_,      // fy: scale factor along vertical axis
            interpolation_flag_ // interpolation method
        );

        // 4. --- Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_local(scaled_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image