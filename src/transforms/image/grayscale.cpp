#include "include/transforms/image/grayscale.h"


// --- Example Main (for testing) ---
// #include "transforms/image/grayscale.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a synthetic color image.
//     cv::Mat image_mat(256, 384, CV_8UC3);
//     // Create three colored bands
//     image_mat(cv::Rect(0, 0, 128, 256)).setTo(cv::Scalar(255, 0, 0));       // Blue
//     image_mat(cv::Rect(128, 0, 128, 256)).setTo(cv::Scalar(0, 255, 0));    // Green
//     image_mat(cv::Rect(256, 0, 128, 256)).setTo(cv::Scalar(0, 0, 255));    // Red
//
//     cv::imwrite("grayscale_before.png", image_mat);
//     std::cout << "Saved grayscale_before.png" << std::endl;
//
//     torch::Tensor image = xt::utils::image::mat_to_tensor_float(image_mat);
//
//     // --- Example 1: Convert to 3-channel grayscale ---
//     std::cout << "\n--- Applying Grayscale (3 channels) ---" << std::endl;
//     xt::transforms::image::Grayscale grayscaler_3ch(3);
//     torch::Tensor gray_3ch_tensor = std::any_cast<torch::Tensor>(grayscaler_3ch.forward({image}));
//     cv::imwrite("grayscale_3ch_after.png", xt::utils::image::tensor_to_mat_8u(gray_3ch_tensor));
//     std::cout << "Saved grayscale_3ch_after.png (Shape: " << gray_3ch_tensor.sizes() << ")" << std::endl;
//
//     // --- Example 2: Convert to 1-channel grayscale ---
//     std::cout << "\n--- Applying Grayscale (1 channel) ---" << std::endl;
//     xt::transforms::image::Grayscale grayscaler_1ch(1);
//     torch::Tensor gray_1ch_tensor = std::any_cast<torch::Tensor>(grayscaler_1ch.forward({image}));
//     // To save a 1-channel tensor, we need to repeat it to 3 channels for cv::imwrite
//     cv::Mat gray_1ch_mat = xt::utils::image::tensor_to_mat_8u(gray_1ch_tensor.repeat({3, 1, 1}));
//     cv::imwrite("grayscale_1ch_after.png", gray_1ch_mat);
//     std::cout << "Saved grayscale_1ch_after.png (Shape: " << gray_1ch_tensor.sizes() << ")" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    Grayscale::Grayscale() : Grayscale(3) {}

    Grayscale::Grayscale(int num_output_channels)
        : num_output_channels_(num_output_channels) {
        if (num_output_channels_ != 1 && num_output_channels_ != 3) {
            throw std::invalid_argument("Number of output channels must be 1 or 3.");
        }
    }

    auto Grayscale::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Grayscale::forward received an empty list.");
        }
        torch::Tensor img = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!img.defined()) {
            throw std::invalid_argument("Input tensor passed to Grayscale is not defined.");
        }

        // If the image is already single-channel, handle appropriately
        if (img.size(0) == 1) {
            if (num_output_channels_ == 1) {
                return img; // Already 1-channel, do nothing
            } else {
                return img.repeat({3, 1, 1}); // Convert 1-channel to 3-channel
            }
        }

        if (img.size(0) != 3) {
            throw std::invalid_argument("Grayscale input must have 1 or 3 channels.");
        }

        // --- Apply Grayscale Conversion ---
        // Use the standard luminance weights (ITU-R BT.601) for RGB conversion.
        // Y = 0.299 * R + 0.587 * G + 0.114 * B
        torch::Tensor weights = torch::tensor({0.299, 0.587, 0.114}, img.options())
                                      .view({3, 1, 1});

        // Perform a weighted sum across the channel dimension (dim 0).
        torch::Tensor grayscale_channel = (img * weights).sum(0, /*keepdim=*/true);

        if (num_output_channels_ == 1) {
            return grayscale_channel;
        } else { // num_output_channels_ == 3
            // Repeat the single channel 3 times to create a [3, H, W] tensor.
            return grayscale_channel.repeat({3, 1, 1});
        }
    }

} // namespace xt::transforms::image