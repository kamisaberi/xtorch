#include <transforms/image/crop.h>


// #include "transforms/image/crop.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor of size [3, 100, 200]
//     torch::Tensor image = torch::randn({3, 100, 200});
//
//     // 2. Define the crop parameters
//     int top = 10;
//     int left = 20;
//     int height = 50;
//     int width = 80;
//
//     // 3. Instantiate the Crop transform
//     xt::transforms::image::Crop cropper(top, left, height, width);
//
//     // 4. Apply the transform
//     std::any result_any = cropper.forward({image});
//     torch::Tensor cropped_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 5. Check the output
//     std::cout << "Original image shape: " << image.sizes() << std::endl;
//     std::cout << "Cropped image shape: " << cropped_image.sizes() << std::endl;
//     // Expected output: [3, 50, 80]
//
//     // --- Example of invalid crop ---
//     try {
//         xt::transforms::image::Crop invalid_cropper(90, 150, 50, 80);
//         invalid_cropper.forward({image});
//     } catch (const std::out_of_range& e) {
//         std::cerr << "\nCaught expected exception: " << e.what() << std::endl;
//     }
//
//     return 0;
// }

namespace xt::transforms::image {

    Crop::Crop() : top_(0), left_(0), height_(-1), width_(-1) {} // Uninitialized state

    Crop::Crop(int top, int left, int height, int width)
        : top_(top), left_(left), height_(height), width_(width) {

        if (height <= 0 || width <= 0) {
            throw std::invalid_argument("Crop height and width must be positive.");
        }
    }

    auto Crop::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Crop::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to Crop is not defined.");
        }
        if (input_tensor.dim() != 3) {
            throw std::invalid_argument("Crop expects a 3D image tensor (C, H, W).");
        }

        const int64_t img_h = input_tensor.size(1);
        const int64_t img_w = input_tensor.size(2);

        // 2. --- Crop Bounds Validation ---
        if (top_ < 0 || left_ < 0 || top_ + height_ > img_h || left_ + width_ > img_w) {
            throw std::out_of_range("Crop dimensions are out of the image bounds.");
        }

        // 3. --- Perform the Crop using Tensor Slicing ---
        // .slice(dimension, start, end)
        // We slice along dimension 1 (height) and dimension 2 (width).
        torch::Tensor cropped_tensor = input_tensor
            .slice(/*dim=*/1, /*start=*/top_, /*end=*/top_ + height_)
            .slice(/*dim=*/2, /*start=*/left_, /*end=*/left_ + width_);

        // The result of slice is a view. To make it a standalone tensor with its own memory,
        // it's good practice to clone it, especially if it's passed to other functions.
        return cropped_tensor.clone();
    }

} // namespace xt::transforms::image