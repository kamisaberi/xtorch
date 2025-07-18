#include <transforms/image/posterize.h>

// #include "transforms/image/posterize.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor with a gradient to see the effect clearly.
//     torch::Tensor image = torch::linspace(0, 1, 256).view({1, -1}).repeat({3, 256, 1});
//
//     // --- Example 1: Strong Posterization (2 bits) ---
//     // This will result in only 2^2 = 4 color levels per channel.
//     std::cout << "--- Posterizing to 2 bits ---" << std::endl;
//     xt::transforms::image::Posterize posterizer_2_bits(2);
//
//     torch::Tensor posterized_2 = std::any_cast<torch::Tensor>(posterizer_2_bits.forward({image}));
//
//     // Find the unique color values in the output
//     torch::Tensor unique_values_2 = std::get<0>(torch::unique_consecutive(posterized_2.flatten()));
//     std::cout << "Number of unique values (2 bits): " << unique_values_2.size(0) << std::endl;
//     std::cout << "Unique values: " << unique_values_2 << std::endl;
//
//     // --- Example 2: Mild Posterization (5 bits) ---
//     // This will result in 2^5 = 32 color levels per channel.
//     std::cout << "\n--- Posterizing to 5 bits ---" << std::endl;
//     xt::transforms::image::Posterize posterizer_5_bits(5);
//
//     torch::Tensor posterized_5 = std::any_cast<torch::Tensor>(posterizer_5_bits.forward({image}));
//
//     torch::Tensor unique_values_5 = std::get<0>(torch::unique_consecutive(posterized_5.flatten()));
//     std::cout << "Number of unique values (5 bits): " << unique_values_5.size(0) << std::endl;
//
//     // You could save the posterized gradient image to see the distinct color bands.
//     // cv::Mat output_mat = xt::utils::image::tensor_to_mat_8u(posterized_2);
//     // cv::imwrite("posterized_image.png", output_mat);
//
//     return 0;
// }

namespace xt::transforms::image {

    Posterize::Posterize() : bits_(4) {}

    Posterize::Posterize(int bits) : bits_(bits) {
        if (bits_ < 1 || bits_ > 8) {
            throw std::invalid_argument("Posterize bits must be between 1 and 8.");
        }
    }

    auto Posterize::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Posterize::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to Posterize is not defined.");
        }

        // If we are keeping all 8 bits, there is no change to the image.
        if (bits_ == 8) {
            return input_tensor;
        }

        // 2. --- Convert to OpenCV Mat (8-bit) ---
        // This operation is most efficiently done on 8-bit integer images.
        cv::Mat input_mat_8u = xt::utils::image::tensor_to_mat_8u(input_tensor);

        // 3. --- Apply Posterization using Bitwise Operations ---
        // Create a bitmask to zero out the lower bits.
        // For example, if bits=4, we want to keep the 4 most significant bits.
        // The mask will be 11110000 in binary, which is 240 in decimal.
        uchar mask = ~((1 << (8 - bits_)) - 1);

        // Use OpenCV's element-wise bitwise AND operation.
        // This operation is very fast as it can be vectorized.
        cv::Mat posterized_mat;
        cv::bitwise_and(input_mat_8u, cv::Scalar::all(mask), posterized_mat);

        // 4. --- Convert back to LibTorch Tensor (Float) ---
        // Convert the 8-bit result back to a float tensor in the [0, 1] range.
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_float(posterized_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image