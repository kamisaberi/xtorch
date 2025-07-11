#include "include/transforms/image/longest_max_size.h"


// #include "transforms/image/longest_max_size.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a large dummy image tensor
//     torch::Tensor large_image = torch::rand({3, 1200, 1600}); // A 1600x1200 landscape image
//
//     // 2. Create a small dummy image tensor
//     torch::Tensor small_image = torch::rand({3, 600, 800});
//
//     // 3. Instantiate the transform with a max size of 1024
//     xt::transforms::image::LongestMaxSize resizer(1024);
//
//     // --- Example 1: Applying to the large image ---
//     std::any result_large_any = resizer.forward({large_image});
//     torch::Tensor resized_large_image = std::any_cast<torch::Tensor>(result_large_any);
//
//     std::cout << "--- Large Image ---" << std::endl;
//     std::cout << "Original shape: " << large_image.sizes() << std::endl;
//     std::cout << "Resized shape:  " << resized_large_image.sizes() << std::endl;
//     // Original longest side is 1600. Scale = 1024/1600 = 0.64.
//     // New W = 1600 * 0.64 = 1024. New H = 1200 * 0.64 = 768.
//     // Expected output: [3, 768, 1024]
//
//     // --- Example 2: Applying to the small image ---
//     std::any result_small_any = resizer.forward({small_image});
//     torch::Tensor resized_small_image = std::any_cast<torch::Tensor>(result_small_any);
//
//     std::cout << "\n--- Small Image ---" << std::endl;
//     std::cout << "Original shape: " << small_image.sizes() << std::endl;
//     std::cout << "Resized shape:  " << resized_small_image.sizes() << std::endl;
//     // Longest side (800) is <= 1024, so no resize happens.
//     // Expected output: [3, 600, 800]
//
//     return 0;
// }

namespace xt::transforms::image {

    LongestMaxSize::LongestMaxSize() : max_size_(1024), interpolation_flag_(cv::INTER_AREA) {}

    LongestMaxSize::LongestMaxSize(int max_size, const std::string& interpolation)
        : max_size_(max_size) {

        if (max_size_ <= 0) {
            throw std::invalid_argument("LongestMaxSize max_size must be a positive integer.");
        }

        if (interpolation == "area") {
            interpolation_flag_ = cv::INTER_AREA;
        } else if (interpolation == "linear") {
            interpolation_flag_ = cv::INTER_LINEAR;
        } else if (interpolation == "cubic") {
            interpolation_flag_ = cv::INTER_CUBIC;
        } else {
            throw std::invalid_argument("Unsupported interpolation method for LongestMaxSize.");
        }
    }

    auto LongestMaxSize::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("LongestMaxSize::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined() || input_tensor.dim() != 3) {
            throw std::invalid_argument("LongestMaxSize expects a defined 3D image tensor (C, H, W).");
        }

        const int64_t H = input_tensor.size(1);
        const int64_t W = input_tensor.size(2);

        // 2. --- Determine if resizing is needed ---
        int64_t longest_side = std::max(H, W);

        if (longest_side <= max_size_) {
            // Image is already small enough, return it unchanged.
            return input_tensor;
        }

        // 3. --- Calculate New Dimensions ---
        double scale = static_cast<double>(max_size_) / longest_side;

        int64_t new_H, new_W;
        if (H > W) {
            new_H = max_size_;
            new_W = static_cast<int64_t>(W * scale);
        } else {
            new_W = max_size_;
            new_H = static_cast<int64_t>(H * scale);
        }

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