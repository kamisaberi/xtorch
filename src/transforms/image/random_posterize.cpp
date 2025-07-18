#include <transforms/image/random_posterize.h>


// --- Example Main (for testing) ---
// #include "transforms/image/random_posterize.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a dummy image tensor with a gradient to see the effect clearly.
//     torch::Tensor image = torch::linspace(0, 1, 256).view({1, -1}).repeat({3, 256, 1});
//     cv::imwrite("posterize_before.png", xt::utils::image::tensor_to_mat_8u(image));
//     std::cout << "Saved posterize_before.png" << std::endl;
//
//     std::cout << "--- Applying RandomPosterize ---" << std::endl;
//
//     // 2. Define transform. Randomly posterize to between 1 and 4 bits.
//     //    This will result in a very strong, noticeable effect.
//     //    Use p=1.0 to guarantee it runs.
//     xt::transforms::image::RandomPosterize posterizer({1, 4}, 1.0);
//
//     // 3. Apply the transform
//     torch::Tensor posterized_tensor = std::any_cast<torch::Tensor>(posterizer.forward({image}));
//
//     // 4. Save the result
//     cv::Mat posterized_mat = xt::utils::image::tensor_to_mat_8u(posterized_tensor);
//     cv::imwrite("posterize_after.png", posterized_mat);
//     std::cout << "Saved posterize_after.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomPosterize::RandomPosterize() : RandomPosterize({4, 8}, 0.5) {}

    RandomPosterize::RandomPosterize(std::pair<int, int> bits_range, double p)
        : bits_range_(bits_range), p_(p) {

        if (bits_range_.first < 1 || bits_range_.second > 8 || bits_range_.first > bits_range_.second) {
            throw std::invalid_argument("Posterize bits range must be valid and between 1 and 8.");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }

        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomPosterize::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomPosterize::forward received an empty list.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomPosterize is not defined.");
        }

        // --- Select Random Number of Bits ---
        std::uniform_int_distribution<int> bits_dist(bits_range_.first, bits_range_.second);
        int bits = bits_dist(gen_);

        // If we are keeping all 8 bits, there is no change to the image.
        if (bits == 8) {
            return input_tensor;
        }

        // --- Convert to OpenCV Mat (8-bit) ---
        // This operation is most efficiently done on 8-bit integer images.
        cv::Mat input_mat_8u = xt::utils::image::tensor_to_mat_8u(input_tensor);

        // --- Apply Posterization using Bitwise Operations ---
        // Create a bitmask to zero out the lower bits.
        // For example, if bits=4, we want to keep the 4 most significant bits.
        // The mask will be 11110000 in binary, which is 240 in decimal.
        uchar mask = ~((1 << (8 - bits)) - 1);

        // Use OpenCV's element-wise bitwise AND operation.
        cv::Mat posterized_mat;
        cv::bitwise_and(input_mat_8u, cv::Scalar::all(mask), posterized_mat);

        // --- Convert back to LibTorch Tensor (Float) ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_float(posterized_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image