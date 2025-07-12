#include "include/transforms/image/random_grayscale.h"


// --- Example Main (for testing) ---
// #include "transforms/image/random_grayscale.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a synthetic color image to see the effect clearly.
//     torch::Tensor image = torch::zeros({3, 200, 200});
//     // Red channel gets a vertical stripe
//     image.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice(80, 120)}, 1.0);
//     // Green channel gets a horizontal stripe
//     image.index_put_({1, torch::indexing::Slice(80, 120), torch::indexing::Slice()}, 1.0);
//     // Blue channel is a constant background
//     image.index_put_({2, torch::indexing::Slice(), torch::indexing::Slice()}, 0.3);
//
//     cv::imwrite("grayscale_before.png", xt::utils::image::tensor_to_mat_8u(image));
//     std::cout << "Saved grayscale_before.png" << std::endl;
//
//     std::cout << "--- Applying RandomGrayscale ---" << std::endl;
//
//     // 2. Define the transform, with p=1.0 to guarantee it runs.
//     xt::transforms::image::RandomGrayscale grayscaler(1.0);
//
//     // 3. Apply the transform
//     torch::Tensor grayscaled_tensor = std::any_cast<torch::Tensor>(grayscaler.forward({image}));
//
//     // 4. Save the result.
//     cv::Mat grayscaled_mat = xt::utils::image::tensor_to_mat_8u(grayscaled_tensor);
//     cv::imwrite("grayscale_after.png", grayscaled_mat);
//     std::cout << "Saved grayscale_after.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomGrayscale::RandomGrayscale() : RandomGrayscale(0.1) {}

    RandomGrayscale::RandomGrayscale(double p) : p_(p) {
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }
        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomGrayscale::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomGrayscale::forward received an empty list.");
        }
        torch::Tensor img = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!img.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomGrayscale is not defined.");
        }

        // If the image is not 3-channel (e.g., already grayscale), do nothing.
        if (img.size(0) != 3) {
            return img;
        }

        // --- Apply Grayscale Conversion ---
        // Use the standard luminance weights (ITU-R BT.601) for RGB conversion.
        // Y = 0.299 * R + 0.587 * G + 0.114 * B
        torch::Tensor weights = torch::tensor({0.299, 0.587, 0.114}, img.options())
                                      .view({3, 1, 1});

        // Perform a weighted sum across the channel dimension (dim 0).
        // The result is a single-channel [1, H, W] tensor.
        torch::Tensor grayscale_channel = (img * weights).sum(0, /*keepdim=*/true);

        // Repeat the single channel 3 times to create a [3, H, W] tensor.
        return grayscale_channel.repeat({3, 1, 1});
    }

} // namespace xt::transforms::image