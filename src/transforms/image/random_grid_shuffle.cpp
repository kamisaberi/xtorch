#include "include/transforms/image/random_grid_shuffle.h"


// --- Example Main (for testing) ---
// #include "transforms/image/random_grid_shuffle.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a sample image that makes shuffling obvious.
//     // A grid with numbers in each cell.
//     cv::Mat image_mat = cv::Mat(256, 256, CV_8UC3, cv::Scalar(255, 255, 255));
//     int grid_size = 4;
//     int cell_size = 256 / grid_size;
//     for (int i = 0; i < grid_size; ++i) {
//         for (int j = 0; j < grid_size; ++j) {
//             int number = i * grid_size + j;
//             cv::Point text_pos(j * cell_size + cell_size / 3, i * cell_size + cell_size / 2);
//             cv::putText(image_mat, std::to_string(number), text_pos, cv::FONT_HERSHEY_SIMPLEX, 1, {0,0,0}, 2);
//             cv::rectangle(image_mat, {j * cell_size, i * cell_size}, {(j + 1) * cell_size, (i + 1) * cell_size}, {128,128,128});
//         }
//     }
//     cv::imwrite("grid_shuffle_before.png", image_mat);
//     std::cout << "Saved grid_shuffle_before.png" << std::endl;
//
//     torch::Tensor image = xt::utils::image::mat_to_tensor_float(image_mat);
//
//     std::cout << "--- Applying RandomGridShuffle (4x4) ---" << std::endl;
//
//     // 2. Define the transform.
//     xt::transforms::image::RandomGridShuffle shuffler(grid_size, 1.0);
//
//     // 3. Apply the transform
//     torch::Tensor shuffled_tensor = std::any_cast<torch::Tensor>(shuffler.forward({image}));
//
//     // 4. Save the result.
//     cv::Mat shuffled_mat = xt::utils::image::tensor_to_mat_8u(shuffled_tensor);
//     cv::imwrite("grid_shuffle_after.png", shuffled_mat);
//     std::cout << "Saved grid_shuffle_after.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomGridShuffle::RandomGridShuffle() : RandomGridShuffle(4, 0.5) {}

    RandomGridShuffle::RandomGridShuffle(int grid_size, double p)
        : grid_size_(grid_size), p_(p) {

        if (grid_size_ < 1) {
            throw std::invalid_argument("Grid size must be at least 1.");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }

        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomGridShuffle::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_ || grid_size_ == 1) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomGridShuffle::forward received an empty list.");
        }
        torch::Tensor img = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!img.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomGridShuffle is not defined.");
        }

        auto h = img.size(1);
        auto w = img.size(2);

        // Calculate cell height and width
        auto cell_h = h / grid_size_;
        auto cell_w = w / grid_size_;

        if (cell_h == 0 || cell_w == 0) {
            // Grid size is too large for the image, do nothing.
            return img;
        }

        // --- Reshape image into a "batch" of grid cells ---
        // 1. Crop the image to be divisible by the grid size
        auto cropped_h = cell_h * grid_size_;
        auto cropped_w = cell_w * grid_size_;
        torch::Tensor cropped_img = img.slice(1, 0, cropped_h).slice(2, 0, cropped_w);

        // 2. Reshape into (C, num_cells_h, cell_h, num_cells_w, cell_w)
        torch::Tensor reshaped = cropped_img.view({
            img.size(0),
            grid_size_,
            cell_h,
            grid_size_,
            cell_w
        });

        // 3. Permute to (num_cells_h, num_cells_w, C, cell_h, cell_w)
        reshaped = reshaped.permute({1, 3, 0, 2, 4}).contiguous();

        // 4. Flatten the grid dimensions to get a batch of cells: (num_cells, C, cell_h, cell_w)
        torch::Tensor cells = reshaped.view({grid_size_ * grid_size_, img.size(0), cell_h, cell_w});

        // --- Shuffle the cells ---
        // 1. Create a shuffled index
        torch::Tensor indices = torch::randperm(cells.size(0), torch::kLong);

        // 2. Index the batch of cells with the shuffled index
        torch::Tensor shuffled_cells = cells.index({indices});

        // --- Reassemble the image from the shuffled cells ---
        // 1. Reshape back into a grid: (num_cells_h, num_cells_w, C, cell_h, cell_w)
        reshaped = shuffled_cells.view({grid_size_, grid_size_, img.size(0), cell_h, cell_w});

        // 2. Permute back to (C, num_cells_h, cell_h, num_cells_w, cell_w)
        reshaped = reshaped.permute({2, 0, 3, 1, 4}).contiguous();

        // 3. Reshape back to the original (cropped) image shape: (C, H', W')
        torch::Tensor shuffled_img = reshaped.view({img.size(0), cropped_h, cropped_w});

        // --- Handle original image size ---
        // If the original image was cropped, we need to paste the shuffled part back
        // into a tensor of the original size.
        if (cropped_h != h || cropped_w != w) {
            torch::Tensor final_img = img.clone();
            final_img.slice(1, 0, cropped_h).slice(2, 0, cropped_w) = shuffled_img;
            return final_img;
        } else {
            return shuffled_img;
        }
    }

} // namespace xt::transforms::image