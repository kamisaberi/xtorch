#include "include/transforms/image/channel_dropout.h"


// #include "transforms/image/channel_dropout.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy 3-channel image
//     // Let's make each channel a solid color to see the effect clearly
//     torch::Tensor R = torch::ones({1, 5, 5}) * 0.2; // Red channel
//     torch::Tensor G = torch::ones({1, 5, 5}) * 0.5; // Green channel
//     torch::Tensor B = torch::ones({1, 5, 5}) * 0.8; // Blue channel
//     torch::Tensor image = torch::cat({R, G, B}, 0);
//
//     std::cout << "Original Image (sum of each channel):" << std::endl;
//     std::cout << image.sum({1, 2}) << std::endl;
//
//     // 2. Instantiate the transform with a 50% chance of dropping a channel
//     xt::transforms::image::ChannelDropout dropper(0.5);
//
//     // 3. Apply the transform multiple times to see different outcomes
//     for (int i = 0; i < 5; ++i) {
//         std::cout << "\n--- Iteration " << i + 1 << " ---" << std::endl;
//         std::any result_any = dropper.forward({image});
//         torch::Tensor dropped_image = std::any_cast<torch::Tensor>(result_any);
//
//         std::cout << "Dropped Image (sum of each channel):" << std::endl;
//         // The sum will be 0 for any channel that was dropped
//         std::cout << dropped_image.sum({1, 2}) << std::endl;
//     }
//
//     return 0;
// }

namespace xt::transforms::image {

    ChannelDropout::ChannelDropout() : p_(0.5) {}

    ChannelDropout::ChannelDropout(double p) : p_(p) {
        if (p < 0.0 || p > 1.0) {
            throw std::invalid_argument("ChannelDropout probability must be between 0 and 1.");
        }
    }

    auto ChannelDropout::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("ChannelDropout::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to ChannelDropout is not defined.");
        }
        if (input_tensor.dim() != 3) {
            throw std::invalid_argument("ChannelDropout expects a 3D image tensor (C, H, W).");
        }

        // If p is 0, do nothing. If p is 1, return a zero tensor.
        if (p_ == 0.0) {
            return input_tensor;
        }
        if (p_ == 1.0) {
            return torch::zeros_like(input_tensor);
        }

        // 2. --- Create the Channel Mask ---
        int64_t num_channels = input_tensor.size(0);

        // Create a random tensor with the same number of elements as there are channels.
        // The values will be uniform random numbers between 0 and 1.
        torch::Tensor channel_rand = torch::rand({num_channels}, input_tensor.options());

        // Create a boolean mask: true where the random number is > p, false otherwise.
        // `true` means keep the channel, `false` means drop it.
        torch::Tensor keep_mask = (channel_rand > p_);

        // 3. --- Reshape and Apply the Mask ---
        // Reshape the mask to [C, 1, 1] so it can be broadcasted correctly for multiplication.
        // The mask will then be multiplied element-wise with the input [C, H, W].
        keep_mask = keep_mask.view({num_channels, 1, 1});

        // Multiply the input tensor by the mask.
        // The mask is automatically converted from boolean (true/false) to the tensor's
        // float/integer type (1.0/0.0) during the multiplication.
        torch::Tensor output_tensor = input_tensor * keep_mask;

        // Optional: Rescale the output to maintain the same expected value.
        // This is standard practice in dropout layers.
        // output_tensor = output_tensor / (1.0 - p_);

        return output_tensor;
    }

} // namespace xt::transforms::image