#include <transforms/image/grid_mask.h>


// #include "transforms/image/grid_mask.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor (e.g., a solid color image)
//     torch::Tensor image = torch::ones({3, 224, 224});
//
//     // 2. Instantiate the transform
//     // Grid cycle size will be between 96 and 224 pixels.
//     // The drop ratio is 0.6, so in a 100x100 cell, a 60x60 square is dropped.
//     // The grid can be rotated up to 30 degrees.
//     // Apply the transform every time for this demo (p=1.0).
//     xt::transforms::image::GridMask masker(
//         /*d_min=*/96,
//         /*d_max=*/224,
//         /*ratio=*/0.6,
//         /*rotate=*/30.0,
//         /*p=*/1.0
//     );
//
//     // 3. Apply the transform
//     std::any result_any = masker.forward({image});
//     torch::Tensor masked_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Original image shape: " << image.sizes() << std::endl;
//     std::cout << "Image with GridMask shape: " << masked_image.sizes() << std::endl;
//
//     // The mean value should be significantly lower than 1.0
//     std::cout << "Original mean value: " << image.mean().item<float>() << std::endl;
//     std::cout << "GridMask mean value: " << masked_image.mean().item<float>() << std::endl;
//
//     // You could save the output image to see the rotated grid of black squares.
//     // cv::Mat output_mat = xt::utils::image::tensor_to_mat_8u(masked_image);
//     // cv::imwrite("gridmask_image.png", output_mat);
//
//     return 0;
// }

namespace xt::transforms::image {

    GridMask::GridMask() : d_min_(96), d_max_(224), ratio_(0.6), rotate_(0.0), p_(0.5) {}

    GridMask::GridMask(int d_min, int d_max, double ratio, double rotate, double p)
        : d_min_(d_min), d_max_(d_max), ratio_(ratio), rotate_(rotate), p_(p) {

        if (p < 0.0 || p > 1.0) {
            throw std::invalid_argument("GridMask probability must be between 0.0 and 1.0.");
        }
        if (ratio < 0.0 || ratio > 1.0) {
            throw std::invalid_argument("GridMask ratio must be between 0.0 and 1.0.");
        }
    }

    auto GridMask::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Decide whether to apply the transform ---
        if (torch::rand({1}).item<float>() > p_) {
            return tensors.begin()[0];
        }

        // --- 2. Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("GridMask::forward received an empty list of tensors.");
        }
        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!image.defined() || image.dim() != 3) {
            throw std::invalid_argument("GridMask expects a defined 3D image tensor (C, H, W).");
        }

        const int64_t H = image.size(1);
        const int64_t W = image.size(2);

        // --- 3. Determine Grid Parameters for this application ---
        int d = torch::randint(d_min_, d_max_ + 1, {1}).item<int>();
        int l = static_cast<int>(d * ratio_); // Size of the dropped square

        // Random starting offset for the grid
        int x_offset = torch::randint(0, d, {1}).item<int>();
        int y_offset = torch::randint(0, d, {1}).item<int>();

        // Random rotation angle
        double angle_rad = torch::rand({1}).item<double>() * rotate_ * M_PI / 180.0;
        double c = std::cos(angle_rad);
        double s = std::sin(angle_rad);

        // --- 4. Generate the Grid Mask ---
        // Create a coordinate grid for all pixels in the image
        auto x_coords = torch::arange(W).repeat({H, 1});
        auto y_coords = torch::arange(H).view({-1, 1}).repeat({1, W});

        // Rotate the coordinates around the center of the image
        auto cx = static_cast<float>(W) / 2;
        auto cy = static_cast<float>(H) / 2;

        auto rot_x = c * (x_coords.to(torch::kFloat32) - cx) - s * (y_coords.to(torch::kFloat32) - cy) + cx;
        auto rot_y = s * (x_coords.to(torch::kFloat32) - cx) + c * (y_coords.to(torch::kFloat32) - cy) + cy;

        // Apply the grid logic to the rotated coordinates
        // A pixel is dropped if its rotated (x,y) falls within the dropped square of a grid cell
        auto grid_x = (rot_x + x_offset).to(torch::kLong) % d;
        auto grid_y = (rot_y + y_offset).to(torch::kLong) % d;

        // `mask` is true where pixels should be KEPT, and false where they should be DROPPED
        torch::Tensor mask = (grid_x >= l) | (grid_y >= l);

        // --- 5. Apply the Mask ---
        // Reshape mask to [1, H, W] so it can be broadcasted across the channels
        mask = mask.to(image.dtype()).unsqueeze(0);

        return image * mask;
    }

} // namespace xt::transforms::image