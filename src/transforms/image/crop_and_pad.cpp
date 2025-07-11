#include "include/transforms/image/crop_and_pad.h"



// #include "transforms/image/crop_and_pad.h"
// #include <iostream>
//
// void print_shape(const std::string& name, const torch::Tensor& t) {
//     std::cout << name << " shape: " << t.sizes() << std::endl;
// }
//
// int main() {
//     // 1. Create a dummy image tensor of size [3, 100, 100]
//     torch::Tensor image = torch::ones({3, 100, 100});
//
//     // --- Example 1: A crop fully inside the image (should behave like regular Crop) ---
//     std::cout << "--- Example 1: Standard Crop ---" << std::endl;
//     xt::transforms::image::CropAndPad standard_cropper(10, 10, 50, 50);
//     torch::Tensor standard_crop = std::any_cast<torch::Tensor>(standard_cropper.forward({image}));
//     print_shape("Standard crop", standard_crop); // Expected: [3, 50, 50]
//
//     // --- Example 2: Crop extending off the right and bottom edges ---
//     std::cout << "\n--- Example 2: Crop off Right/Bottom Edge ---" << std::endl;
//     // We want a 50x50 crop starting at (80, 80).
//     // This will grab a 20x20 piece from the image and pad it to 50x50.
//     xt::transforms::image::CropAndPad edge_cropper(80, 80, 50, 50);
//     torch::Tensor edge_crop = std::any_cast<torch::Tensor>(edge_cropper.forward({image}));
//     print_shape("Edge crop", edge_crop); // Expected: [3, 50, 50]
//
//     // --- Example 3: Crop extending off the top and left edges ---
//     std::cout << "\n--- Example 3: Crop off Top/Left Edge ---" << std::endl;
//     // We want a 50x50 crop starting at (-20, -20).
//     // This will grab a 30x30 piece from the image and pad it to 50x50.
//     xt::transforms::image::CropAndPad negative_cropper(-20, -20, 50, 50, 0.5f);
//     torch::Tensor negative_crop = std::any_cast<torch::Tensor>(negative_cropper.forward({image}));
//     print_shape("Negative crop", negative_crop); // Expected: [3, 50, 50]
//
//     // --- Example 4: Crop completely outside the image ---
//     std::cout << "\n--- Example 4: Crop Fully Outside ---" << std::endl;
//     // We want a 50x50 crop starting at (200, 200).
//     // This will create a 50x50 padded image.
//     xt::transforms::image::CropAndPad outside_cropper(200, 200, 50, 50);
//     torch::Tensor outside_crop = std::any_cast<torch::Tensor>(outside_cropper.forward({image}));
//     print_shape("Outside crop", outside_crop); // Expected: [3, 50, 50]
//     std::cout << "Mean value of outside crop: " << outside_crop.mean().item<float>() << std::endl; // Should be 0 (the fill_value)
//
//
//     return 0;
// }

namespace xt::transforms::image {

    CropAndPad::CropAndPad() : top_(0), left_(0), height_(-1), width_(-1), fill_value_(0.0f) {}

    CropAndPad::CropAndPad(int top, int left, int height, int width, float fill_value)
        : top_(top), left_(left), height_(height), width_(width), fill_value_(fill_value) {

        if (height <= 0 || width <= 0) {
            throw std::invalid_argument("Target height and width must be positive.");
        }
    }

    auto CropAndPad::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("CropAndPad::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined() || input_tensor.dim() != 3) {
            throw std::invalid_argument("CropAndPad expects a defined 3D image tensor (C, H, W).");
        }

        const int64_t img_h = input_tensor.size(1);
        const int64_t img_w = input_tensor.size(2);

        // 2. --- Calculate the Intersection (the part we can actually crop) ---
        int64_t crop_top = std::max(0, top_);
        int64_t crop_left = std::max(0, left_);
        int64_t crop_bottom = std::min(img_h, (int64_t)top_ + height_);
        int64_t crop_right = std::min(img_w, (int64_t)left_ + width_);

        // Crop the valid region. If the region is empty, this creates an empty tensor.
        torch::Tensor cropped_tensor = input_tensor.slice(1, crop_top, crop_bottom)
                                                   .slice(2, crop_left, crop_right);

        // 3. --- Calculate Required Padding ---
        // Padding is needed if the crop started before the image (e.g., top_ < 0)
        // or ended after the image.
        int64_t pad_top = std::max(0, -top_);
        int64_t pad_left = std::max(0, -left_);

        // Calculate how much padding is needed on the bottom and right to reach the target size.
        int64_t cropped_h = crop_bottom - crop_top;
        int64_t cropped_w = crop_right - crop_left;

        int64_t pad_bottom = height_ - (cropped_h + pad_top);
        int64_t pad_right = width_ - (cropped_w + pad_left);

        // Ensure padding is not negative (can happen in edge cases)
        pad_bottom = std::max((int64_t)0, pad_bottom);
        pad_right = std::max((int64_t)0, pad_right);

        // 4. --- Apply Padding if Necessary ---
        torch::nn::functional::PadFuncOptions padding_options({pad_left, pad_right, pad_top, pad_bottom});
        padding_options.mode(torch::kConstant);
        padding_options.value(fill_value_);

        torch::Tensor output_tensor = torch::nn::functional::pad(cropped_tensor, padding_options);

        return output_tensor;
    }

} // namespace xt::transforms::image