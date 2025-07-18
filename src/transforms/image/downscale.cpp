#include <transforms/image/downscale.h>

#include "transforms/image/downscale.h"
#include <iostream>

// int main() {
//     // 1. Create a dummy image tensor of size [3, 200, 300]
//     torch::Tensor image = torch::rand({3, 200, 300});
//
//     // 2. Instantiate the transform to scale the image to 50% of its size
//     xt::transforms::image::Downscale downscaler(0.5);
//
//     // 3. Apply the transform
//     std::any result_any = downscaler.forward({image});
//     torch::Tensor downscaled_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Original image shape: " << image.sizes() << std::endl;
//     std::cout << "Downscaled image shape: " << downscaled_image.sizes() << std::endl;
//     // Expected output: [3, 100, 150]
//
//     // --- Example with a different factor ---
//     xt::transforms::image::Downscale downscaler_quarter(0.25);
//     torch::Tensor quarter_image = std::any_cast<torch::Tensor>(downscaler_quarter.forward({image}));
//     std::cout << "\nQuarter-sized image shape: " << quarter_image.sizes() << std::endl;
//     // Expected output: [3, 50, 75]
//
//     return 0;
// }

namespace xt::transforms::image {

    Downscale::Downscale() : scale_factor_(0.5), interpolation_flag_(cv::INTER_AREA) {}

    Downscale::Downscale(double scale_factor, const std::string& interpolation)
        : scale_factor_(scale_factor) {

        if (scale_factor <= 0.0 || scale_factor > 1.0) {
            throw std::invalid_argument("Downscale scale_factor must be between 0.0 and 1.0.");
        }

        if (interpolation == "area") {
            interpolation_flag_ = cv::INTER_AREA;
        } else if (interpolation == "linear") {
            interpolation_flag_ = cv::INTER_LINEAR;
        } else if (interpolation == "cubic") {
            interpolation_flag_ = cv::INTER_CUBIC;
        } else {
            throw std::invalid_argument("Unsupported interpolation method for Downscale.");
        }
    }

    auto Downscale::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Downscale::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to Downscale is not defined.");
        }

        // 2. --- Convert to OpenCV Mat ---
        // We use your existing conversion utility.
        cv::Mat input_mat = xt::utils::image::tensor_to_mat_local(input_tensor);

        // 3. --- Apply Downscaling using cv::resize ---
        cv::Mat downscaled_mat;

        // When fx and fy are specified, dsize should be (0,0)
        cv::resize(
            input_mat,          // source image
            downscaled_mat,     // destination image
            cv::Size(),         // target size (0,0 to use scale factors)
            scale_factor_,      // fx: scale factor along horizontal axis
            scale_factor_,      // fy: scale factor along vertical axis
            interpolation_flag_ // interpolation method
        );

        // 4. --- Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_local(downscaled_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image