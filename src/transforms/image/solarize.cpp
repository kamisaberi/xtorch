#include <transforms/image/solarize.h>

// #include "transforms/image/solarize.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor with a gradient from black to white
//     //    to clearly see the solarization effect.
//     torch::Tensor image = torch::linspace(0, 1, 256).view({1, -1}).repeat({3, 100, 1});
//     // Now image has shape [3, 100, 256] with a horizontal gradient.
//
//     // 2. Instantiate the transform with a threshold of 0.5
//     xt::transforms::image::Solarize solarizer(0.5f);
//
//     // 3. Apply the transform
//     std::any result_any = solarizer.forward({image});
//     torch::Tensor solarized_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Image shapes are unchanged." << std::endl;
//     std::cout << "Original image shape:    " << image.sizes() << std::endl;
//     std::cout << "Solarized image shape:   " << solarized_image.sizes() << std::endl;
//
//     // Check a pixel value below the threshold
//     float val_below_thresh = solarized_image.index({0, 50, 50}).item<float>();
//     std::cout << "Value at a point < 0.5 is unchanged: " << val_below_thresh << std::endl;
//
//     // Check a pixel value above the threshold
//     float original_val_above = image.index({0, 50, 200}).item<float>();
//     float solarized_val_above = solarized_image.index({0, 50, 200}).item<float>();
//     std::cout << "Value at a point > 0.5 is inverted: " << original_val_above
//               << " -> " << solarized_val_above << std::endl;
//     // Expected result: ~0.78 -> ~0.22
//
//
//     // You could save the output image to see the effect.
//     // The gradient will go from black to white, then suddenly flip and go from black back to gray.
//     // cv::Mat output_mat = xt::utils::image::tensor_to_mat_8u(solarized_image);
//     // cv::imwrite("solarized_image.png", output_mat);
//
//     return 0;
// }

namespace xt::transforms::image {

    Solarize::Solarize() : threshold_(0.5f) {}

    Solarize::Solarize(float threshold) : threshold_(threshold) {}

    auto Solarize::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Solarize::forward received an empty list of tensors.");
        }
        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!image.defined()) {
            throw std::invalid_argument("Input tensor passed to Solarize is not defined.");
        }

        // 2. --- Apply Solarization ---
        // Create a boolean mask where `true` indicates a pixel to be inverted.
        torch::Tensor invert_mask = (image > threshold_);

        // Invert the original image completely
        // Assumes the image is normalized in the [0, 1] range.
        torch::Tensor inverted_image = 1.0f - image;

        // The `torch::where` function is perfect for this.
        // It takes a condition (our mask) and two tensors.
        // It returns a new tensor where elements are taken from `inverted_image`
        // if the condition is true, and from the original `image` if false.
        torch::Tensor solarized_image = torch::where(invert_mask, inverted_image, image);

        return solarized_image;
    }

} // namespace xt::transforms::image