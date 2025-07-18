#include <transforms/image/crop_non_empty_mask_if_exists.h>





// #include "transforms/image/crop_non_empty_mask_if_exists.h"
// #include <iostream>
//
// void print_shape(const std::string& name, const torch::Tensor& t) {
//     std::cout << name << " shape: " << t.sizes() << std::endl;
// }
//
// int main() {
//     // Target crop size
//     int height = 50;
//     int width = 50;
//
//     // Instantiate the transform
//     xt::transforms::image::CropNonEmptyMaskIfExists cropper(height, width);
//
//     // --- Example 1: Only an image is passed ---
//     std::cout << "--- Example 1: Image only (Random Crop) ---" << std::endl;
//     torch::Tensor image_only = torch::randn({3, 100, 100});
//
//     std::any result1 = cropper.forward({image_only});
//     torch::Tensor cropped_image1 = std::any_cast<torch::Tensor>(result1);
//
//     print_shape("Original Image", image_only);
//     print_shape("Cropped Image", cropped_image1); // Expected: [3, 50, 50]
//
//     // --- Example 2: Image and Mask are passed ---
//     std::cout << "\n--- Example 2: Image and Mask ---" << std::endl;
//     torch::Tensor image_with_mask = torch::randn({3, 100, 100});
//     torch::Tensor mask = torch::zeros({1, 100, 100}, torch::kInt8);
//     // Create a non-empty region in the mask from (20,30) to (40, 80)
//     mask.slice(1, 20, 40).slice(2, 30, 80) = 1;
//
//     std::any result2 = cropper.forward({image_with_mask, mask});
//     auto result_pair = std::any_cast<std::pair<torch::Tensor, torch::Tensor>>(result2);
//     torch::Tensor cropped_image2 = result_pair.first;
//     torch::Tensor cropped_mask2 = result_pair.second;
//
//     print_shape("Original Image", image_with_mask);
//     print_shape("Original Mask", mask);
//     print_shape("Cropped Image", cropped_image2); // Expected: [3, 50, 50]
//     print_shape("Cropped Mask", cropped_mask2);   // Expected: [1, 50, 50]
//
//     // The cropped mask should still contain non-zero values
//     std::cout << "Sum of cropped mask: " << cropped_mask2.sum().item<int64_t>() << std::endl;
//
//     return 0;
// }

namespace xt::transforms::image {

    CropNonEmptyMaskIfExists::CropNonEmptyMaskIfExists() : height_(-1), width_(-1) {}

    CropNonEmptyMaskIfExists::CropNonEmptyMaskIfExists(int height, int width)
        : height_(height), width_(width) {
        if (height <= 0 || width <= 0) {
            throw std::invalid_argument("Crop height and width must be positive.");
        }
    }

    // Private helper for the fallback random crop
    auto CropNonEmptyMaskIfExists::_random_crop(torch::Tensor& img) -> torch::Tensor {
        const int64_t img_h = img.size(1);
        const int64_t img_w = img.size(2);

        if (height_ > img_h || width_ > img_w) {
             throw std::invalid_argument("Random crop size cannot be larger than the image size.");
        }

        int64_t top = torch::randint(0, img_h - height_ + 1, {1}).item<int64_t>();
        int64_t left = torch::randint(0, img_w - width_ + 1, {1}).item<int64_t>();

        return img.slice(1, top, top + height_).slice(2, left, left + width_).clone();
    }

    auto CropNonEmptyMaskIfExists::forward(std::initializer_list<std::any> tensors) -> std::any {
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("CropNonEmptyMaskIfExists::forward received an empty list of tensors.");
        }

        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]);
        if (!image.defined() || image.dim() != 3) {
            throw std::invalid_argument("Input image must be a defined 3D tensor (C, H, W).");
        }

        // --- Case 1: No mask provided, perform a random crop ---
        if (any_vec.size() == 1) {
            return _random_crop(image);
        }

        // --- Case 2: Mask is provided ---
        torch::Tensor mask = std::any_cast<torch::Tensor>(any_vec[1]);
        if (!mask.defined() || mask.dim() < 2 || mask.dim() > 3) {
             throw std::invalid_argument("Input mask must be a defined 2D or 3D tensor.");
        }
        if (mask.size(-2) != image.size(-2) || mask.size(-1) != image.size(-1)) {
            throw std::invalid_argument("Image and mask must have the same height and width.");
        }

        // Find the coordinates of all non-zero pixels in the mask
        torch::Tensor non_zero_indices = (mask > 0).nonzero();

        // If the mask is empty, fallback to a random crop of the image and return a zero mask
        if (non_zero_indices.size(0) == 0) {
            torch::Tensor cropped_image = _random_crop(image);
            // Create a corresponding empty mask of the target size
            torch::Tensor cropped_mask = torch::zeros({1, height_, width_}, mask.options());
            return std::make_pair(cropped_image, cropped_mask);
        }

        // The indices tensor has shape [N, Dims], where Dims is 2 (for HW) or 3 (for CHW)
        // We are interested in the last two dimensions (height and width)
        torch::Tensor y_coords = non_zero_indices.select(1, -2);
        torch::Tensor x_coords = non_zero_indices.select(1, -1);

        // Find the bounding box
        int64_t y_min = y_coords.min().item<int64_t>();
        int64_t y_max = y_coords.max().item<int64_t>();
        int64_t x_min = x_coords.min().item<int64_t>();
        int64_t x_max = x_coords.max().item<int64_t>();

        // Crop both the image and the mask using this bounding box
        auto cropped_image = image.slice(1, y_min, y_max + 1).slice(2, x_min, x_max + 1);
        auto cropped_mask = mask.slice(-2, y_min, y_max + 1).slice(-1, x_min, x_max + 1);

        // Resize both crops to the target size
        // unsqueeze(0) adds a batch dimension for interpolate
        auto resize_options = torch::nn::functional::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({height_, width_}));

        auto final_image = torch::nn::functional::interpolate(
            cropped_image.unsqueeze(0),
            resize_options.mode(torch::kBilinear).align_corners(false)
        ).squeeze(0);

        auto final_mask = torch::nn::functional::interpolate(
            cropped_mask.to(torch::kFloat).unsqueeze(0),
            resize_options.mode(torch::kNearest) // Use nearest neighbor for masks to preserve labels
        ).squeeze(0).to(mask.scalar_type());

        return std::make_pair(final_image, final_mask);
    }

} // namespace xt::transforms::image