#include <transforms/image/smallest_max_size.h>

// #include "transforms/image/smallest_max_size.h"
// #include <iostream>
//
// int main() {
//     // Target size for the smallest side
//     int target_size = 256;
//
//     // Instantiate the transform
//     xt::transforms::image::SmallestMaxSize resizer(target_size, "cubic");
//
//     // --- Example 1: A landscape image (width > height) ---
//     torch::Tensor landscape_image = torch::rand({3, 480, 640});
//     // Smallest side is height (480).
//     // Scale = 256 / 480 ~= 0.5333
//     // New H = 480 * 0.5333 = 256
//     // New W = 640 * 0.5333 = 341
//
//     std::cout << "--- Landscape Image ---" << std::endl;
//     std::cout << "Original shape: " << landscape_image.sizes() << std::endl;
//     torch::Tensor resized1 = std::any_cast<torch::Tensor>(resizer.forward({landscape_image}));
//     std::cout << "Resized shape:  " << resized1.sizes() << std::endl;
//     // Expected output: [3, 256, 341]
//
//
//     // --- Example 2: A portrait image (height > width) ---
//     torch::Tensor portrait_image = torch::rand({3, 1080, 720});
//     // Smallest side is width (720).
//     // Scale = 256 / 720 ~= 0.3555
//     // New W = 720 * 0.3555 = 256
//     // New H = 1080 * 0.3555 = 384
//
//     std::cout << "\n--- Portrait Image ---" << std::endl;
//     std::cout << "Original shape: " << portrait_image.sizes() << std::endl;
//     torch::Tensor resized2 = std::any_cast<torch::Tensor>(resizer.forward({portrait_image}));
//     std::cout << "Resized shape:  " << resized2.sizes() << std::endl;
//     // Expected output: [3, 384, 256]
//
//     return 0;
// }

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