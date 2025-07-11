#include "include/transforms/image/mask_dropout.h"

// #include "transforms/image/mask_dropout.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image and a mask
//     torch::Tensor image = torch::ones({3, 100, 100}); // A solid white image
//     torch::Tensor mask = torch::zeros({1, 100, 100}, torch::kInt8);
//     // Create a circular foreground mask in the center
//     auto y = torch::arange(100).view({-1, 1});
//     auto x = torch::arange(100);
//     auto dist_from_center = torch::sqrt(torch::pow(y - 50, 2) + torch::pow(x - 50, 2));
//     mask.masked_fill_(dist_from_center < 30, 1);
//
//     std::cout << "Original image mean: " << image.mean().item<float>() << std::endl;
//
//     // 2. Instantiate the transform
//     // Drop 30% of the pixels within the mask, filling with a value of 0.2 (dark gray).
//     // Apply this with a 100% probability for the demo.
//     xt::transforms::image::MaskDropout dropper(
//         /*max_objects=*/1,
//         /*p=*/1.0f,
//         /*holes_nb=*/1,
//         /*mask_fill_value=*/0,
//         /*p_replace=*/0.3f
//     );
//
//     // 3. Apply the transform
//     std::any result_any = dropper.forward({image, mask});
//     torch::Tensor dropped_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Image with mask dropout shape: " << dropped_image.sizes() << std::endl;
//     // The mean value should be lower than the original's 1.0, because we dropped pixels.
//     std::cout << "Image with mask dropout mean: " << dropped_image.mean().item<float>() << std::endl;
//
//     // You could save the output image to see the "peppered" holes inside the
//     // circular region, while the background remains untouched.
//     // cv::Mat output_mat = xt::utils::image::tensor_to_mat_8u(dropped_image);
//     // cv::imwrite("mask_dropout_image.png", output_mat);
//
//     return 0;
// }

namespace xt::transforms::image {

    MaskDropout::MaskDropout()
        : max_objects_(1), p_(0.5f), holes_nb_(1), mask_fill_value_(0), p_replace_(0.1f) {}

    MaskDropout::MaskDropout(int max_objects, float p, int holes_nb, int mask_fill_value, float p_replace)
        : max_objects_(max_objects), p_(p), holes_nb_(holes_nb), mask_fill_value_(mask_fill_value), p_replace_(p_replace) {

        if (p_ < 0.0f || p_ > 1.0f) {
            throw std::invalid_argument("MaskDropout probability `p` must be between 0.0 and 1.0.");
        }
    }

    auto MaskDropout::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Decide whether to apply the transform ---
        if (torch::rand({1}).item<float>() > p_) {
            return tensors.begin()[0]; // Return the original image
        }

        // --- 2. Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("MaskDropout::forward expects two tensors: image and mask.");
        }
        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]).clone(); // Clone to modify
        torch::Tensor mask = std::any_cast<torch::Tensor>(any_vec[1]);

        if (!image.defined() || !mask.defined()) {
            throw std::invalid_argument("Input image or mask is not defined.");
        }
        if (image.dim() != 3) {
            throw std::invalid_argument("Input image must be a 3D tensor [C, H, W].");
        }
        if (mask.dim() < 2 || mask.dim() > 3) {
            throw std::invalid_argument("Input mask must be a 2D or 3D tensor.");
        }
        if (image.size(-2) != mask.size(-2) || image.size(-1) != mask.size(-1)) {
            throw std::invalid_argument("Image and mask must have the same height and width.");
        }

        // Ensure mask is a boolean tensor [H, W] for easier processing
        torch::Tensor bool_mask = (mask > 0).squeeze();

        // --- 3. Find All Foreground Pixels ---
        torch::Tensor foreground_indices = bool_mask.nonzero();
        if (foreground_indices.size(0) == 0) {
            return image; // No foreground pixels to drop, return original image
        }

        // --- 4. Select Pixels to Drop ---
        int64_t num_foreground_pixels = foreground_indices.size(0);
        int64_t num_pixels_to_drop = static_cast<int64_t>(num_foreground_pixels * p_replace_);

        if (num_pixels_to_drop == 0) {
            return image; // Nothing to drop
        }

        // Create a random permutation of the foreground pixel indices
        torch::Tensor rand_perm = torch::randperm(num_foreground_pixels, image.options().dtype(torch::kLong));
        // Select the first `num_pixels_to_drop` indices from the permutation
        torch::Tensor indices_to_drop = foreground_indices.index_select(0, rand_perm.slice(0, 0, num_pixels_to_drop));

        // --- 5. Apply the Dropout ---
        // The indices tensor is [N, 2], where each row is a (y, x) coordinate.
        // We need to use advanced indexing to set the values at these coordinates to the fill value.
        // We can do this by splitting the y and x coordinates.
        auto y_coords = indices_to_drop.select(1, 0);
        auto x_coords = indices_to_drop.select(1, 1);

        // For each channel in the image, apply the mask.
        for (int c = 0; c < image.size(0); ++c) {
            image.index_put_({torch::indexing::None, y_coords, x_coords}, mask_fill_value_);
        }

        // A more "pytorchy" way without a loop, but slightly more complex:
        // We create a full mask for all channels.
        // auto drop_mask = torch::zeros_like(image, torch::kBool);
        // drop_mask.index_put_({torch::indexing::Slice(), y_coords, x_coords}, true);
        // image.masked_fill_(drop_mask, mask_fill_value_);

        return image;
    }

} // namespace xt::transforms::image