#include <transforms/image/random_scale.h>


// --- Example Main (for testing) ---
// #include "transforms/image/random_scale.h"
// #include "utils/image_conversion.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
//
// int main() {
//     // 1. Create a sample image.
//     cv::Mat image_mat(200, 300, CV_8UC3, cv::Scalar(255, 255, 255));
//     // Draw a circle to visualize the scaling.
//     cv::circle(image_mat, {150, 100}, 80, {255, 0, 0}, -1);
//     cv::imwrite("scale_before.png", image_mat);
//     std::cout << "Saved scale_before.png (Shape: " << image_mat.size() << ")" << std::endl;
//
//     torch::Tensor image = xt::utils::image::mat_to_tensor_float(image_mat);
//
//     std::cout << "--- Applying RandomScale (Downsampling) ---" << std::endl;
//
//     // 2. Define transform to downsample (scale factor < 1.0)
//     xt::transforms::image::RandomScale downsampler({0.4, 0.6});
//     torch::Tensor downsampled_tensor = std::any_cast<torch::Tensor>(downsampler.forward({image}));
//     cv::Mat downsampled_mat = xt::utils::image::tensor_to_mat_8u(downsampled_tensor);
//     cv::imwrite("scale_downsampled.png", downsampled_mat);
//     std::cout << "Saved scale_downsampled.png (Shape: " << downsampled_mat.size() << ")" << std::endl;
//
//
//     std::cout << "\n--- Applying RandomScale (Upsampling) ---" << std::endl;
//
//     // 3. Define transform to upsample (scale factor > 1.0)
//     xt::transforms::image::RandomScale upsampler({1.5, 2.0});
//     torch::Tensor upsampled_tensor = std::any_cast<torch::Tensor>(upsampler.forward({image}));
//     cv::Mat upsampled_mat = xt::utils::image::tensor_to_mat_8u(upsampled_tensor);
//     cv::imwrite("scale_upsampled.png", upsampled_mat);
//     std::cout << "Saved scale_upsampled.png (Shape: " << upsampled_mat.size() << ")" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomScale::RandomScale() : RandomScale({0.8, 1.2}) {}

    RandomScale::RandomScale(
        std::pair<double, double> scale_range,
        const std::string& interpolation)
        : scale_range_(scale_range), interpolation_mode_str_(interpolation) {

        // --- Parameter Validation ---
        if (scale_range_.first <= 0 || scale_range_.second <= 0 || scale_range_.first > scale_range_.second) {
            throw std::invalid_argument("Scale range must be valid and positive.");
        }

        if (interpolation_mode_str_ != "bilinear" &&
            interpolation_mode_str_ != "nearest" &&
            interpolation_mode_str_ != "bicubic") {
            throw std::invalid_argument("Unsupported interpolation type.");
        }

        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomScale::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomScale::forward received an empty list.");
        }
        torch::Tensor img = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!img.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomScale is not defined.");
        }

        // --- Determine Random Scale Factor ---
        std::uniform_real_distribution<> scale_dist(scale_range_.first, scale_range_.second);
        double scale_factor = scale_dist(gen_);

        // If factor is 1.0, no change is needed (small optimization).
        if (std::abs(scale_factor - 1.0) < 1e-6) {
            return img;
        }

        // --- Calculate New Size ---
        auto h = img.size(1);
        auto w = img.size(2);
        int new_h = static_cast<int>(std::round(h * scale_factor));
        int new_w = static_cast<int>(std::round(w * scale_factor));

        if (new_h == 0 || new_w == 0) {
            // Avoid creating a zero-sized image
            return img;
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