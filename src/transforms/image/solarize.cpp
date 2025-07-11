#include "include/transforms/image/solarize.h"


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