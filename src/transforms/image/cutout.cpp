#include <transforms/image/cutout.h>


// --- Example Main (for testing) ---
// #include "transforms/image/cutout.h"
// #include "utils/image_conversion.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
//
// int main() {
//     // 1. Create a sample image.
//     cv::Mat image_mat(256, 256, CV_8UC3);
//     // Create a colorful gradient
//     for (int i = 0; i < image_mat.rows; ++i) {
//         for (int j = 0; j < image_mat.cols; ++j) {
//             image_mat.at<cv::Vec3b>(i, j) = cv::Vec3b(i, j, (i + j) % 255);
//         }
//     }
//     cv::imwrite("cutout_before.png", image_mat);
//     std::cout << "Saved cutout_before.png" << std::endl;
//
//     torch::Tensor image = xt::utils::image::mat_to_tensor_float(image_mat);
//
//     std::cout << "--- Applying Cutout ---" << std::endl;
//
//     // 2. Define transform: Create 10 holes of size 32x32, filled with gray.
//     xt::transforms::image::Cutout cutter(
//         /*num_holes=*/10,
//         /*size=*/{32, 32},
//         /*fill=*/0.5,
//         /*p=*/1.0
//     );
//
//     // 3. Apply the transform
//     torch::Tensor cut_tensor = std::any_cast<torch::Tensor>(cutter.forward({image}));
//
//     // 4. Save the result
//     cv::Mat cut_mat = xt::utils::image::tensor_to_mat_8u(cut_tensor);
//     cv::imwrite("cutout_after.png", cut_mat);
//     std::cout << "Saved cutout_after.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    Cutout::Cutout() : Cutout(1, {16, 16}) {}

    Cutout::Cutout(
        int num_holes,
        std::pair<int, int> size,
        double fill,
        double p)
        : num_holes_(num_holes), size_(size), fill_(fill), p_(p) {

        if (num_holes_ < 1) {
            throw std::invalid_argument("Number of holes must be at least 1.");
        }
        if (size_.first <= 0 || size_.second <= 0) {
            throw std::invalid_argument("Cutout size must be positive.");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }

        std::random_device rd;
        gen_.seed(rd());
    }

    auto Cutout::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Cutout::forward received an empty list.");
        }
        torch::Tensor img = std::any_cast<torch::Tensor>(any_vec[0]).clone(); // Clone to modify in-place

        if (!img.defined()) {
            throw std::invalid_argument("Input tensor passed to Cutout is not defined.");
        }

        auto h = img.size(1);
        auto w = img.size(2);
        auto hole_h = size_.first;
        auto hole_w = size_.second;

        for (int n = 0; n < num_holes_; ++n) {
            // --- Select a random center for the hole ---
            std::uniform_int_distribution<int> y_dist(0, h - 1);
            std::uniform_int_distribution<int> x_dist(0, w - 1);
            auto center_y = y_dist(gen_);
            auto center_x = x_dist(gen_);

            // --- Calculate hole boundaries and clamp them to the image ---
            int64_t y1 = std::max((int64_t)0, (int64_t)center_y - hole_h / 2);
            int64_t y2 = std::min((int64_t)h, (int64_t)center_y + hole_h / 2);
            int64_t x1 = std::max((int64_t)0, (int64_t)center_x - hole_w / 2);
            int64_t x2 = std::min((int64_t)w, (int64_t)center_x + hole_w / 2);

            // --- Fill the patch ---
            // Use advanced indexing to set the region to the fill value.
            if (y2 > y1 && x2 > x1) {
                img.index_put_({torch::indexing::Slice(),
                                 torch::indexing::Slice(y1, y2),
                                 torch::indexing::Slice(x1, x2)}, fill_);
            }
        }

        return img;
    }

} // namespace xt::transforms::image