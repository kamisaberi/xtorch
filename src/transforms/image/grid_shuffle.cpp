#include "include/transforms/image/grid_shuffle.h"

// #include "transforms/image/grid_shuffle.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor that's easy to visualize.
//     // Let's create a 2x2 grid where each quadrant is a different color.
//     // Image size is 200x200, so each quadrant is 100x100.
//     torch::Tensor image = torch::zeros({3, 200, 200});
//     image.slice(1, 0, 100).slice(2, 0, 100).index_put_({0}, 1.0);     // Top-left: Red
//     image.slice(1, 0, 100).slice(2, 100, 200).index_put_({1}, 1.0);  // Top-right: Green
//     image.slice(1, 100, 200).slice(2, 0, 100).index_put_({2}, 1.0);  // Bottom-left: Blue
//     image.slice(1, 100, 200).slice(2, 100, 200) = 0.5;                // Bottom-right: Gray
//
//     // You could save the original image to see the starting state.
//     // cv::Mat original_mat = xt::utils::image::tensor_to_mat_8u(image);
//     // cv::imwrite("original_quadrants.png", original_mat);
//
//     // 2. Instantiate the transform with a 2x2 grid size.
//     // This will shuffle the four colored quadrants.
//     xt::transforms::image::GridShuffle shuffler(2);
//
//     // 3. Apply the transform
//     std::any result_any = shuffler.forward({image});
//     torch::Tensor shuffled_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Original image shape: " << image.sizes() << std::endl;
//     std::cout << "Shuffled image shape: " << shuffled_image.sizes() << std::endl;
//
//     // You can now save the shuffled image. The colored quadrants will be in a new, random order.
//     // cv::Mat shuffled_mat = xt::utils::image::tensor_to_mat_8u(shuffled_image);
//     // cv::imwrite("shuffled_quadrants.png", shuffled_mat);
//
//     return 0;
// }

namespace xt::transforms::image {

    GridShuffle::GridShuffle() : grid_size_(4) {}

    GridShuffle::GridShuffle(int grid_size) : grid_size_(grid_size) {
        if (grid_size_ <= 1) {
            throw std::invalid_argument("GridShuffle grid_size must be greater than 1.");
        }
    }

    auto GridShuffle::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("GridShuffle::forward received an empty list of tensors.");
        }
        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!image.defined() || image.dim() != 3) {
            throw std::invalid_argument("GridShuffle expects a defined 3D image tensor (C, H, W).");
        }

        const int64_t C = image.size(0);
        const int64_t H = image.size(1);
        const int64_t W = image.size(2);

        // Ensure the image can be evenly divided by the grid size
        if (H % grid_size_ != 0 || W % grid_size_ != 0) {
            // A more robust implementation could pad the image first, but for now, we'll require compatibility.
            throw std::invalid_argument("Image height and width must be divisible by the grid_size.");
        }

        // 2. --- Reshape image into grid cells ---
        const int64_t cell_h = H / grid_size_;
        const int64_t cell_w = W / grid_size_;

        // Reshape the tensor to isolate the grid cells.
        // The idea is to transform [C, H, W] -> [C, grid_h, cell_h, grid_w, cell_w]
        torch::Tensor reshaped = image.view({C, grid_size_, cell_h, grid_size_, cell_w});

        // 3. --- Shuffle the grid cells ---
        // Permute the dimensions to group the grid cells together.
        // [C, grid_h, cell_h, grid_w, cell_w] -> [grid_h, grid_w, C, cell_h, cell_w]
        reshaped = reshaped.permute({1, 3, 0, 2, 4}).contiguous();

        // Now, the first two dimensions represent the grid (grid_size_ x grid_size_).
        // We can flatten these two dimensions to get a list of all cells.
        // Shape becomes: [grid_size*grid_size, C, cell_h, cell_w]
        reshaped = reshaped.view({grid_size_ * grid_size_, C, cell_h, cell_w});

        // Create random permutation of the cell indices.
        torch::Tensor rand_indices = torch::randperm(grid_size_ * grid_size_, image.options().dtype(torch::kLong));

        // Apply the shuffle.
        reshaped = reshaped.index_select(0, rand_indices);

        // 4. --- Reassemble the image from the shuffled cells ---
        // Reshape back to the grid format [grid_h, grid_w, C, cell_h, cell_w]
        reshaped = reshaped.view({grid_size_, grid_size_, C, cell_h, cell_w});

        // Permute back to the original logical order before the final reshape.
        // [grid_h, grid_w, C, cell_h, cell_w] -> [C, grid_h, cell_h, grid_w, cell_w]
        reshaped = reshaped.permute({2, 0, 3, 1, 4}).contiguous();

        // Finally, reshape back to the original image shape [C, H, W]
        torch::Tensor shuffled_image = reshaped.view({C, H, W});

        return shuffled_image;
    }

} // namespace xt::transforms::image