#include <transforms/image/channel_shuffle.h>

// #include "transforms/image/channel_shuffle.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor with 6 channels for easy shuffling into 2 or 3 groups.
//     // Let's make the channels sequential numbers to easily see the shuffle.
//     // Shape: [6, 2, 2]
//     torch::Tensor image = torch::arange(0, 6 * 2 * 2, torch::kFloat32).view({6, 2, 2});
//     std::cout << "Original Image (showing first element of each channel):\n"
//               << image.slice(1, 0, 1).slice(2, 0, 1).flatten() << std::endl;
//     // Expected output: [ 0.,  4.,  8., 12., 16., 20.]
//
//     // --- Example 1: Shuffle with 2 groups ---
//     // Channels [0, 1, 2] are group 1. Channels [3, 4, 5] are group 2.
//     // After shuffle, channels should be mixed, e.g., [0, 3, 1, 4, 2, 5].
//     std::cout << "\n--- Shuffling with 2 groups ---" << std::endl;
//     xt::transforms::image::ChannelShuffle shuffler_2_groups(2);
//
//     std::any result_any = shuffler_2_groups.forward({image});
//     torch::Tensor shuffled_image = std::any_cast<torch::Tensor>(result_any);
//
//     std::cout << "Shuffled Image (showing first element of each channel):\n"
//               << shuffled_image.slice(1, 0, 1).slice(2, 0, 1).flatten() << std::endl;
//     // Expected output: [ 0., 12.,  4., 16.,  8., 20.] - Channels from group 2 are interleaved.
//
//
//     // --- Example 2: Shuffle with 3 groups ---
//     // Groups: [0, 1], [2, 3], [4, 5]
//     // After shuffle, channels should be mixed, e.g., [0, 2, 4, 1, 3, 5]
//     std::cout << "\n--- Shuffling with 3 groups ---" << std::endl;
//     xt::transforms::image::ChannelShuffle shuffler_3_groups(3);
//
//     result_any = shuffler_3_groups.forward({image});
//     shuffled_image = std::any_cast<torch::Tensor>(result_any);
//
//     std::cout << "Shuffled Image (showing first element of each channel):\n"
//               << shuffled_image.slice(1, 0, 1).slice(2, 0, 1).flatten() << std::endl;
//     // Expected output: [ 0.,  8., 16.,  4., 12., 20.] - Channels from groups 2 and 3 are interleaved.
//
//     return 0;
// }

namespace xt::transforms::image {

    ChannelShuffle::ChannelShuffle() : groups_(2) {}

    ChannelShuffle::ChannelShuffle(int groups) : groups_(groups) {
        if (groups_ <= 0) {
            throw std::invalid_argument("ChannelShuffle groups must be a positive integer.");
        }
    }

    auto ChannelShuffle::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("ChannelShuffle::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to ChannelShuffle is not defined.");
        }

        // This transform can work on batches or single images, but let's be explicit
        // and assume a single image (3D) or a batch (4D) for robustness.
        int dims = input_tensor.dim();
        if (dims < 3) {
            throw std::invalid_argument("ChannelShuffle expects at least a 3D tensor (C, H, W).");
        }

        int64_t num_channels, height, width, batch_size;
        bool is_batched = (dims == 4);

        if (is_batched) {
            batch_size = input_tensor.size(0);
            num_channels = input_tensor.size(1);
            height = input_tensor.size(2);
            width = input_tensor.size(3);
        } else { // 3D tensor
            batch_size = 1; // Treat as a batch of 1 for unified logic
            num_channels = input_tensor.size(0);
            height = input_tensor.size(1);
            width = input_tensor.size(2);
            input_tensor = input_tensor.unsqueeze(0); // Add batch dimension
        }

        if (num_channels % groups_ != 0) {
            throw std::invalid_argument("Number of channels must be divisible by the number of groups.");
        }

        // 2. --- Perform the Shuffle ---
        int64_t channels_per_group = num_channels / groups_;

        // Step A: Reshape into groups
        // [B, C, H, W] -> [B, G, N, H, W] where N = channels_per_group
        torch::Tensor x = input_tensor.view({batch_size, groups_, channels_per_group, height, width});

        // Step B: Transpose the group and channel-in-group dimensions
        // [B, G, N, H, W] -> [B, N, G, H, W]
        x = x.transpose(1, 2);

        // Step C: Reshape back to the original channel layout
        // The call to .contiguous() is crucial here. Transposing creates a view of the
        // tensor with a non-contiguous memory layout. Calling .view() on a non-contiguous
        // tensor will throw an error. .contiguous() returns a new tensor with the data
        // laid out in contiguous memory, allowing the final reshape.
        x = x.contiguous().view({batch_size, num_channels, height, width});

        // 3. --- Finalize Output ---
        // If the original input was a single image, remove the batch dimension we added.
        if (!is_batched) {
            x = x.squeeze(0);
        }

        return x;
    }

} // namespace xt::transforms::image