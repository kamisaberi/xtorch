#include <transforms/image/random_short_side_scale.h>


// --- Example Main (for testing) ---
// #include "transforms/image/random_short_side_scale.h"
// #include "utils/image_conversion.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
//
// int main() {
//     // 1. Create a non-square sample image.
//     cv::Mat image_mat(480, 800, CV_8UC3, cv::Scalar(255, 255, 255));
//     // Draw a circle to visualize the scaling.
//     cv::circle(image_mat, {400, 240}, 150, {255, 0, 0}, -1);
//     cv::imwrite("short_side_scale_before.png", image_mat);
//     std::cout << "Saved short_side_scale_before.png (Shape: " << image_mat.size() << ")" << std::endl;
//
//     torch::Tensor image = xt::utils::image::mat_to_tensor_float(image_mat);
//
//     std::cout << "--- Applying RandomShortSideScale ---" << std::endl;
//
//     // 2. Define transform. The shorter side (480) will be scaled to between 600 and 700.
//     //    The longer side (800) will be scaled proportionally.
//     xt::transforms::image::RandomShortSideScale scaler({600, 700});
//     torch::Tensor scaled_tensor = std::any_cast<torch::Tensor>(scaler.forward({image}));
//     cv::Mat scaled_mat = xt::utils::image::tensor_to_mat_8u(scaled_tensor);
//     cv::imwrite("short_side_scale_after.png", scaled_mat);
//     std::cout << "Saved short_side_scale_after.png (Shape: " << scaled_mat.size() << ")" << std::endl;
//
//     std::cout << "\n--- Applying RandomShortSideScale with max_size constraint ---" << std::endl;
//
//     // 3. Define transform with a max_size limit.
//     //    Short side (480) will be scaled to ~800, so long side (800) would become ~1333.
//     //    The max_size=1000 constraint will then downscale both.
//     xt::transforms::image::RandomShortSideScale scaler_max({800, 800}, 1000);
//     torch::Tensor scaled_max_tensor = std::any_cast<torch::Tensor>(scaler_max.forward({image}));
//     cv::Mat scaled_max_mat = xt::utils::image::tensor_to_mat_8u(scaled_max_tensor);
//     cv::imwrite("short_side_scale_max_after.png", scaled_max_mat);
//     std::cout << "Saved short_side_scale_max_after.png (Shape: " << scaled_max_mat.size() << ")" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomShortSideScale::RandomShortSideScale() : RandomShortSideScale({640, 800}) {}

    RandomShortSideScale::RandomShortSideScale(
        std::pair<int, int> short_side_range,
        int max_size,
        const std::string& interpolation)
        : short_side_range_(short_side_range), max_size_(max_size), interpolation_mode_str_(interpolation) {

        // --- Parameter Validation ---
        if (short_side_range_.first <= 0 || short_side_range_.second <= 0 || short_side_range_.first > short_side_range_.second) {
            throw std::invalid_argument("Short side range must be valid and positive.");
        }
        if (max_size_ < 0) {
            throw std::invalid_argument("Max size must be non-negative.");
        }
        if (interpolation_mode_str_ != "bilinear" &&
            interpolation_mode_str_ != "nearest" &&
            interpolation_mode_str_ != "bicubic") {
            throw std::invalid_argument("Unsupported interpolation type.");
        }

        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomShortSideScale::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomShortSideScale::forward received an empty list.");
        }
        torch::Tensor img = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!img.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomShortSideScale is not defined.");
        }

        // --- Calculate New Size ---
        auto h = img.size(1);
        auto w = img.size(2);

        std::uniform_int_distribution<int> size_dist(short_side_range_.first, short_side_range_.second);
        int new_short_side = size_dist(gen_);

        int new_h, new_w;
        if (h < w) { // Height is the shorter side
            new_h = new_short_side;
            new_w = static_cast<int>(std::round((double)new_short_side * (double)w / (double)h));
        } else { // Width is the shorter side
            new_w = new_short_side;
            new_h = static_cast<int>(std::round((double)new_short_side * (double)h / (double)w));
        }

        // --- Apply max_size constraint if necessary ---
        if (max_size_ > 0) {
            int longer_side = std::max(new_h, new_w);
            if (longer_side > max_size_) {
                double scale = (double)max_size_ / (double)longer_side;
                new_h = static_cast<int>(std::round(new_h * scale));
                new_w = static_cast<int>(std::round(new_w * scale));
            }
        }

        if (new_h == h && new_w == w) {
            return img; // No change needed
        }
        if (new_h == 0 || new_w == 0) {
            return img; // Avoid creating a zero-sized image
        }

        // --- Resize the Image ---
        namespace F = torch::nn::functional;

        auto interp_opts = F::InterpolateFuncOptions()
            .size(std::vector<int64_t>{new_h, new_w});

        if (interpolation_mode_str_ == "bilinear") {
            interp_opts.mode(torch::kBilinear).align_corners(false);
        } else if (interpolation_mode_str_ == "bicubic") {
            interp_opts.mode(torch::kBicubic).align_corners(false);
        } else { // "nearest"
            interp_opts.mode(torch::kNearest);
        }

        return F::interpolate(
            img.unsqueeze(0), // interpolate needs a batch dimension
            interp_opts
        ).squeeze(0); // remove the batch dimension
    }

} // namespace xt::transforms::image