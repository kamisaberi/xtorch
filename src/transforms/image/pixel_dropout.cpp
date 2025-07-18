#include <transforms/image/pixel_dropout.h>

// #include "transforms/image/pixel_dropout.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor (a solid gray image)
//     torch::Tensor image = torch::ones({3, 100, 100}) * 0.8;
//
//     std::cout << "Original image mean: " << image.mean().item<float>() << std::endl;
//     // Expected output: 0.8
//
//     // 2. Instantiate the transform to drop 20% of pixels and replace them with 0 (black).
//     xt::transforms::image::PixelDropout dropper(0.20, 0.0f);
//
//     // 3. Apply the transform
//     std::any result_any = dropper.forward({image});
//     torch::Tensor dropped_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Image with pixel dropout shape: " << dropped_image.sizes() << std::endl;
//
//     // The mean should be lower now. Approximately 0.8 * (1 - 0.2) = 0.64
//     std::cout << "Image with pixel dropout mean: " << dropped_image.mean().item<float>() << std::endl;
//
//     // You could save the output image to see the "salt-and-pepper" like noise.
//     // cv::Mat output_mat = xt::utils::image::tensor_to_mat_8u(dropped_image);
//     // cv::imwrite("pixel_dropout_image.png", output_mat);
//
//     return 0;
// }

namespace xt::transforms::image {

    PixelDropout::PixelDropout() : p_(0.05), drop_value_(0.0f) {}

    PixelDropout::PixelDropout(double p, float drop_value) : p_(p), drop_value_(drop_value) {
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("PixelDropout probability must be between 0 and 1.");
        }
    }

    auto PixelDropout::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("PixelDropout::forward received an empty list of tensors.");
        }
        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!image.defined()) {
            throw std::invalid_argument("Input tensor passed to PixelDropout is not defined.");
        }

        // If p is 0, no pixels are dropped, so we can return early.
        if (p_ == 0.0) {
            return image;
        }

        // It's better to work on a clone to not modify the original tensor passed in.
        torch::Tensor noisy_image = image.clone();

        // 2. --- Create the Dropout Mask ---
        // Create a tensor of the same size as the image with random values from [0, 1).
        torch::Tensor random_mask = torch::rand_like(noisy_image);

        // Create a boolean mask where `true` indicates a pixel to be dropped.
        torch::Tensor drop_mask = random_mask < p_;

        // 3. --- Apply the Mask ---
        // `masked_fill_` is an in-place operation that sets the elements of `noisy_image`
        // to `drop_value_` wherever the `drop_mask` is `true`.
        noisy_image.masked_fill_(drop_mask, drop_value_);

        return noisy_image;
    }

} // namespace xt::transforms::image