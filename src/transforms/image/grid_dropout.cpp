#include <transforms/image/grid_dropout.h>


// #include "transforms/image/grid_dropout.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor (e.g., a solid color image)
//     torch::Tensor image = torch::ones({3, 200, 200});
//
//     // 2. Instantiate the transform
//     // Drop 40% of the grid cells. Let the transform calculate the cell size.
//     xt::transforms::image::GridDropout dropper(0.4f);
//
//     // 3. Apply the transform
//     std::any result_any = dropper.forward({image});
//     torch::Tensor dropped_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Original image shape: " << image.sizes() << std::endl;
//     std::cout << "Image with grid dropout shape: " << dropped_image.sizes() << std::endl;
//
//     // The mean value of the output image should be lower than the original's 1.0,
//     // roughly around 1.0 * (1.0 - ratio) = 0.6.
//     std::cout << "Original mean value: " << image.mean().item<float>() << std::endl;
//     std::cout << "GridDropout mean value: " << dropped_image.mean().item<float>() << std::endl;
//
//     // You could now save the image to visually inspect the random grid pattern of dropped-out squares.
//     // cv::Mat output_mat = xt::utils::image::tensor_to_mat_8u(dropped_image);
//     // cv::imwrite("grid_dropout_image.png", output_mat);
//
//     return 0;
// }

namespace xt::transforms::image {

    GridDropout::GridDropout(
        float ratio, int unit_size_min, int unit_size_max,
        int holes_nb_min, int holes_nb_max, float fill_value
    ) : ratio_(ratio), unit_size_min_(unit_size_min), unit_size_max_(unit_size_max),
        holes_nb_min_(holes_nb_min), holes_nb_max_(holes_nb_max), fill_value_(fill_value) {

        if (ratio < 0.0f || ratio > 1.0f) {
            throw std::invalid_argument("GridDropout ratio must be between 0.0 and 1.0.");
        }
    }

    // Default constructor for convenience
    GridDropout::GridDropout() : GridDropout(0.5f) {}

    auto GridDropout::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("GridDropout::forward received an empty list of tensors.");
        }
        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]).clone();

        if (!image.defined() || image.dim() != 3) {
            throw std::invalid_argument("GridDropout expects a defined 3D image tensor (C, H, W).");
        }

        if (ratio_ == 0.0f) {
            return image;
        }

        // 2. --- Determine Grid Parameters ---
        const int64_t img_h = image.size(1);
        const int64_t img_w = image.size(2);

        // Determine the size of each grid cell for this application
        int64_t unit_size;
        if (unit_size_min_ == -1) { // Default logic if not provided
            unit_size = static_cast<int64_t>(std::min(img_h, img_w) / 8.0);
        } else {
            unit_size = torch::randint(unit_size_min_, unit_size_max_ + 1, {1}).item<int64_t>();
        }

        // 3. --- Create the Grid Mask ---
        int64_t grid_h = (img_h + unit_size - 1) / unit_size;
        int64_t grid_w = (img_w + unit_size - 1) / unit_size;

        // Create a small mask for the grid cells
        auto grid_mask = (torch::rand({grid_h, grid_w}) < ratio_).to(image.dtype());

        // Expand the small grid mask to the full image size using kronecker product
        // `torch::kron(A, B)` creates a block matrix where each element of A is
        // multiplied by the entire matrix B.
        auto block = torch::ones({unit_size, unit_size}, image.options());
        auto mask = torch::kron(grid_mask, block);

        // Crop the mask to the exact image size
        mask = mask.slice(0, 0, img_h).slice(1, 0, img_w);

        // 4. --- Apply the Mask ---
        // Reshape mask to [1, H, W] to broadcast across channels
        mask = mask.unsqueeze(0);

        // A value of 1 in the mask means drop the pixel. We multiply the image by (1-mask).
        image = image * (1.0f - mask);

        // If a fill_value other than 0 is desired, add it where the mask is active.
        if (fill_value_ != 0.0f) {
            image += mask * fill_value_;
        }

        return image;
    }

} // namespace xt::transforms::image