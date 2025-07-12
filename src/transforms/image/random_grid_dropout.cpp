#include "include/transforms/image/random_grid_dropout.h"


// --- Example Main (for testing) ---
// #include "transforms/image/random_grid_dropout.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a sample image to see the effect clearly.
//     // A gradient works well.
//     torch::Tensor image = torch::linspace(0, 1, 300 * 400).view({1, 300, 400}).repeat({3, 1, 1});
//     cv::imwrite("grid_dropout_before.png", xt::utils::image::tensor_to_mat_8u(image));
//     std::cout << "Saved grid_dropout_before.png" << std::endl;
//
//     std::cout << "--- Applying RandomGridDropout ---" << std::endl;
//
//     // 2. Define the transform. Grid cycle size between 80 and 120 pixels,
//     //    with a dropout ratio of 0.4 (so holes are smaller than the remaining parts).
//     //    We use p=1.0 to guarantee the transform is applied.
//     xt::transforms::image::RandomGridDropout dropper(0.4, {80, 120}, 0.0, 1.0);
//
//     // 3. Apply the transform
//     torch::Tensor dropped_tensor = std::any_cast<torch::Tensor>(dropper.forward({image}));
//
//     // 4. Save the result.
//     cv::Mat dropped_mat = xt::utils::image::tensor_to_mat_8u(dropped_tensor);
//     cv::imwrite("grid_dropout_after.png", dropped_mat);
//     std::cout << "Saved grid_dropout_after.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomGridDropout::RandomGridDropout() : RandomGridDropout(0.5, {50, 150}, 0.0, 0.5) {}

    RandomGridDropout::RandomGridDropout(
        double ratio,
        std::pair<int, int> grid_size,
        double fill,
        double p)
        : ratio_(ratio), grid_size_(grid_size), fill_(fill), p_(p) {

        if (ratio_ < 0.0 || ratio_ > 1.0) {
            throw std::invalid_argument("Ratio must be between 0.0 and 1.0.");
        }
        if (grid_size_.first <= 0 || grid_size_.second <= 0 || grid_size_.first > grid_size_.second) {
            throw std::invalid_argument("Grid size range must be valid and positive.");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }

        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomGridDropout::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomGridDropout::forward received an empty list.");
        }
        torch::Tensor img = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!img.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomGridDropout is not defined.");
        }

        // No-op if ratio is 0
        if (ratio_ == 0.0) {
            return img;
        }

        auto h = img.size(1);
        auto w = img.size(2);

        // --- Generate Grid Parameters ---
        // 1. Random grid cycle size (d)
        std::uniform_int_distribution<> size_dist(grid_size_.first, grid_size_.second);
        auto d = size_dist(gen_);

        // 2. Grid hole size (l)
        auto l = static_cast<int>(d * ratio_);

        // No-op if hole size is 0
        if (l == 0) {
            return img;
        }

        // 3. Random grid offset (cx, cy)
        std::uniform_int_distribution<> x_offset_dist(0, d - 1);
        std::uniform_int_distribution<> y_offset_dist(0, d - 1);
        auto cx = x_offset_dist(gen_);
        auto cy = y_offset_dist(gen_);

        // --- Create the Grid Mask ---
        // A mask of 1s where pixels should be kept, and 0s where they should be dropped.
        torch::Tensor mask = torch::ones_like(img, torch::kBool);

        // Iterate through the image with grid-sized steps
        for (int i = cy; i < h; i += d) {
            for (int j = cx; j < w; j += d) {
                // Define the top-left and bottom-right corners of the dropout square
                auto y1 = std::max(0L, (long)i);
                auto y2 = std::min((long)h, (long)i + l);
                auto x1 = std::max(0L, (long)j);
                auto x2 = std::min((long)w, (long)j + l);

                // Set the mask in this square region to 0 (false)
                if (y2 > y1 && x2 > x1) {
                    mask.index_put_({torch::indexing::Slice(),
                                     torch::indexing::Slice(y1, y2),
                                     torch::indexing::Slice(x1, x2)}, false);
                }
            }
        }

        // --- Apply the Mask ---
        // Use torch::where to select between original image and fill value.
        // where(condition, value_if_true, value_if_false)
        torch::Tensor fill_tensor = torch::full_like(img, fill_);
        return torch::where(mask, img, fill_tensor);
    }

} // namespace xt::transforms::image