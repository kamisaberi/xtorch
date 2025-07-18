#include <transforms/image/random_equalize.h>


// --- Example Main (for testing) ---
// #include "transforms/image/random_equalize.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a low-contrast image.
//     // A gradient with a very small range, which is a perfect candidate for equalization.
//     torch::Tensor low_contrast_tensor = torch::linspace(0.4, 0.6, 256).view({1, -1}).repeat({3, 256, 1});
//     cv::imwrite("equalize_before.png", xt::utils::image::tensor_to_mat_8u(low_contrast_tensor));
//     std::cout << "Saved equalize_before.png" << std::endl;
//
//     std::cout << "--- Applying RandomEqualize ---" << std::endl;
//
//     // 2. Define the transform, with p=1.0 to guarantee it runs.
//     xt::transforms::image::RandomEqualize equalizer(1.0);
//
//     // 3. Apply the transform
//     torch::Tensor equalized_tensor = std::any_cast<torch::Tensor>(equalizer.forward({low_contrast_tensor}));
//
//     // 4. Save the result. The output should have a much wider range of intensities.
//     cv::Mat equalized_mat = xt::utils::image::tensor_to_mat_8u(equalized_tensor);
//     cv::imwrite("equalize_after.png", equalized_mat);
//     std::cout << "Saved equalize_after.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomEqualize::RandomEqualize() : RandomEqualize(0.5) {}

    RandomEqualize::RandomEqualize(double p) : p_(p) {
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }
        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomEqualize::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomEqualize::forward received an empty list.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomEqualize is not defined.");
        }

        // --- Convert to OpenCV Mat (8-bit) ---
        // Histogram equalization is defined for 8-bit integer images.
        cv::Mat input_mat_8u = xt::utils::image::tensor_to_mat_8u(input_tensor);

        // --- Apply Histogram Equalization Per Channel ---
        // OpenCV's equalizeHist only works on single-channel images.
        // For color images, we must split the channels, equalize each one, and merge them back.

        cv::Mat output_mat_8u;
        if (input_mat_8u.channels() == 1) {
            // Grayscale image
            cv::equalizeHist(input_mat_8u, output_mat_8u);
        } else if (input_mat_8u.channels() == 3) {
            // Color image
            std::vector<cv::Mat> channels;
            cv::split(input_mat_8u, channels);

            for (auto& channel : channels) {
                cv::equalizeHist(channel, channel);
            }

            cv::merge(channels, output_mat_8u);
        } else {
            throw std::invalid_argument("RandomEqualize supports only 1 or 3 channel images.");
        }

        // --- Convert back to LibTorch Tensor (Float) ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_float(output_mat_8u);

        return output_tensor;
    }

} // namespace xt::transforms::image