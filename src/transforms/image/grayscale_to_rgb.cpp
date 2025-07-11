#include "include/transforms/image/grayscale_to_rgb.h"



// #include "transforms/image/grayscale_to_rgb.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy grayscale image tensor of size [1, 100, 100]
//     torch::Tensor grayscale_image = torch::rand({1, 100, 100});
//
//     // 2. Instantiate the transform
//     xt::transforms::image::GrayscaleToRGB converter;
//
//     // 3. Apply the transform
//     std::any result_any = converter.forward({grayscale_image});
//     torch::Tensor rgb_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Original grayscale shape: " << grayscale_image.sizes() << std::endl;
//     std::cout << "Converted RGB shape: " << rgb_image.sizes() << std::endl;
//     // Expected output: [3, 100, 100]
//
//     // Verify that all three channels are identical
//     bool channels_are_identical = torch::all(rgb_image[0] == rgb_image[1]).item<bool>() &&
//                                   torch::all(rgb_image[1] == rgb_image[2]).item<bool>();
//
//     std::cout << "Are all channels identical? " << std::boolalpha << channels_are_identical << std::endl;
//
//     // --- Test with an already-RGB image ---
//     torch::Tensor already_rgb = torch::rand({3, 50, 50});
//     torch::Tensor result_rgb = std::any_cast<torch::Tensor>(converter.forward({already_rgb}));
//     std::cout << "\nPassing an RGB image through... Final shape: " << result_rgb.sizes() << std::endl;
//     // Expected output: [3, 50, 50]
//
//     return 0;
// }

namespace xt::transforms::image {

    GrayscaleToRGB::GrayscaleToRGB() = default;

    auto GrayscaleToRGB::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("GrayscaleToRGB::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to GrayscaleToRGB is not defined.");
        }

        // Check for correct dimensions
        if (input_tensor.dim() != 3) {
            throw std::invalid_argument("GrayscaleToRGB expects a 3D image tensor (C, H, W).");
        }

        // --- Main Logic ---
        int64_t num_channels = input_tensor.size(0);

        if (num_channels == 3) {
            // If the image is already 3-channel, just return it as is.
            return input_tensor;
        }

        if (num_channels != 1) {
            throw std::invalid_argument("GrayscaleToRGB expects a 1-channel or 3-channel image tensor.");
        }

        // 2. --- Convert 1-channel to 3-channel ---
        // Use the `repeat` method to stack the single channel three times along the channel dimension.
        // The arguments to repeat are the number of repetitions for each dimension.
        // We want to repeat 3 times for dim 0 (channels), and 1 time for dims 1 and 2 (H and W).
        return input_tensor.repeat({3, 1, 1});
    }

} // namespace xt::transforms::image