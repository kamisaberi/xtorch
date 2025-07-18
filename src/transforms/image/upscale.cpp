#include <transforms/image/upscale.h>

// #include "transforms/image/upscale.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor of size [3, 100, 150]
//     torch::Tensor image = torch::rand({3, 100, 150});
//
//     std::cout << "Original image shape: " << image.sizes() << std::endl;
//
//     // --- Example 1: Double the size ---
//     xt::transforms::image::Upscale upscaler_2x(2.0, "cubic");
//
//     torch::Tensor upscaled_2x_image = std::any_cast<torch::Tensor>(upscaler_2x.forward({image}));
//
//     std::cout << "\n--- Upscaled Image (2x) ---" << std::endl;
//     std::cout << "Upscaled image shape: " << upscaled_2x_image.sizes() << std::endl;
//     // New H = 100 * 2.0 = 200. New W = 150 * 2.0 = 300.
//     // Expected output: [3, 200, 300]
//
//
//     // --- Example 2: Upscale by 3.5x ---
//     xt::transforms::image::Upscale upscaler_3_5x(3.5, "linear");
//
//     torch::Tensor upscaled_3_5x_image = std::any_cast<torch::Tensor>(upscaler_3_5x.forward({image}));
//
//     std::cout << "\n--- Upscaled Image (3.5x) ---" << std::endl;
//     std::cout << "Upscaled image shape: " << upscaled_3_5x_image.sizes() << std::endl;
//     // New H = 100 * 3.5 = 350. New W = 150 * 3.5 = 525.
//     // Expected output: [3, 350, 525]
//
//     return 0;
// }
namespace xt::transforms::image {

    Upscale::Upscale() : scale_factor_(2.0), interpolation_flag_(cv::INTER_CUBIC) {}

    Upscale::Upscale(double scale_factor, const std::string& interpolation)
        : scale_factor_(scale_factor) {

        if (scale_factor_ < 1.0) {
            throw std::invalid_argument("Upscale scale_factor must be >= 1.0. For downscaling, use the Downscale transform.");
        }

        if (interpolation == "cubic") {
            interpolation_flag_ = cv::INTER_CUBIC;
        } else if (interpolation == "linear") {
            interpolation_flag_ = cv::INTER_LINEAR;
        } else if (interpolation == "nearest") {
            interpolation_flag_ = cv::INTER_NEAREST;
        } else {
            throw std::invalid_argument("Unsupported interpolation method for Upscale. 'cubic' or 'linear' are recommended.");
        }
    }

    auto Upscale::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Upscale::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to Upscale is not defined.");
        }

        // If scale factor is 1.0, no change is needed.
        if (scale_factor_ == 1.0) {
            return input_tensor;
        }

        // 2. --- Convert to OpenCV Mat ---
        cv::Mat input_mat = xt::utils::image::tensor_to_mat_local(input_tensor);

        // 3. --- Apply Upscaling using cv::resize ---
        cv::Mat upscaled_mat;

        // When fx and fy are specified, dsize should be (0,0) for cv::resize
        // to use the scale factors instead of an absolute size.
        cv::resize(
            input_mat,          // source image
            upscaled_mat,       // destination image
            cv::Size(),         // target size (dsize=0 to use scale factors)
            scale_factor_,      // fx: scale factor along horizontal axis
            scale_factor_,      // fy: scale factor along vertical axis
            interpolation_flag_ // interpolation method
        );

        // 4. --- Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_local(upscaled_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image