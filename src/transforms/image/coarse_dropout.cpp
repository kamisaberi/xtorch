#include <transforms/image/coarse_dropout.h>

// #include "transforms/image/coarse_dropout.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor (e.g., a solid color image)
//     torch::Tensor image = torch::ones({3, 100, 100}) * 0.8; // A light gray image
//
//     // 2. Instantiate the transform
//     // Create up to 5 holes, each up to 20% of the image's height/width.
//     // Fill with a value of 0.2 (dark gray).
//     xt::transforms::image::CoarseDropout cutter(
//         /*max_holes=*/5,
//         /*max_height=*/0.2f,
//         /*max_width=*/0.2f,
//         /*min_holes=*/1,
//         /*min_height=*/0.1f, // Make holes at least 10% of height/width
//         /*min_width=*/0.1f,
//         /*fill_value=*/0.2f
//     );
//
//     // 3. Apply the transform
//     std::any result_any = cutter.forward({image});
//     torch::Tensor cutout_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Original image shape: " << image.sizes() << std::endl;
//     std::cout << "Image with cutout shape: " << cutout_image.sizes() << std::endl;
//
//     // You could now save the image to visually inspect the random black boxes.
//     // The mean value of the output image should be lower than the original's 0.8.
//     std::cout << "Original mean value: " << image.mean().item<float>() << std::endl;
//     std::cout << "Cutout mean value: " << cutout_image.mean().item<float>() << std::endl;
//
//     return 0;
// }

namespace xt::transforms::image {

    CoarseDropout::CoarseDropout()
        : max_holes_(8), max_height_(0.1f), max_width_(0.1f),
          min_holes_(1), min_height_(-1.0f), min_width_(-1.0f), fill_value_(0.0f) {}

    CoarseDropout::CoarseDropout(
        int max_holes, float max_height, float max_width,
        int min_holes, float min_height, float min_width, float fill_value
    ) : max_holes_(max_holes), max_height_(max_height), max_width_(max_width),
        min_holes_(min_holes), min_height_(min_height), min_width_(min_width), fill_value_(fill_value) {

        if (min_height_ < 0) min_height_ = max_height_;
        if (min_width_ < 0) min_width_ = max_width_;

        if (max_holes_ < 0 || min_holes_ < 0 || max_height_ <= 0 || max_width_ <= 0 || min_height_ <= 0 || min_width_ <= 0) {
            throw std::invalid_argument("CoarseDropout parameters must be positive.");
        }
        if (min_holes_ > max_holes_ || min_height_ > max_height_ || min_width_ > max_width_) {
            throw std::invalid_argument("Min values cannot be greater than max values in CoarseDropout.");
        }
    }

    auto CoarseDropout::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("CoarseDropout::forward received an empty list of tensors.");
        }
        // Make a clone so we can modify it in-place without affecting the original tensor
        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]).clone();

        if (!image.defined() || image.dim() != 3) {
            throw std::invalid_argument("CoarseDropout expects a defined 3D image tensor (C, H, W).");
        }

        const int64_t img_h = image.size(1);
        const int64_t img_w = image.size(2);

        // 2. --- Determine Number of Holes ---
        int num_holes = torch::randint(min_holes_, max_holes_ + 1, {1}).item<int>();

        // 3. --- Create and Apply Holes ---
        for (int i = 0; i < num_holes; ++i) {
            // Determine hole size
            int64_t hole_h = static_cast<int64_t>(
                torch::rand({1}).item<float>() * (max_height_ - min_height_) * img_h + min_height_ * img_h
            );
            int64_t hole_w = static_cast<int64_t>(
                torch::rand({1}).item<float>() * (max_width_ - min_width_) * img_w + min_width_ * img_w
            );
            hole_h = std::min(hole_h, img_h);
            hole_w = std::min(hole_w, img_w);

            // Determine top-left corner of the hole
            int64_t y1 = torch::randint(0, img_h - hole_h + 1, {1}).item<int64_t>();
            int64_t x1 = torch::randint(0, img_w - hole_w + 1, {1}).item<int64_t>();

            int64_t y2 = y1 + hole_h;
            int64_t x2 = x1 + hole_w;

            // Use slicing to select the rectangular region across all channels
            // and fill it with the desired value.
            image.slice(/*dim=*/1, y1, y2).slice(/*dim=*/2, x1, x2) = fill_value_;
        }

        return image;
    }

} // namespace xt::transforms::image