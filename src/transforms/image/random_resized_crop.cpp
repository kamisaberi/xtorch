#include <transforms/image/random_resized_crop.h>


// --- Example Main (for testing) ---
// #include "transforms/image/random_resized_crop.h"
// #include "utils/image_conversion.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
//
// int main() {
//     // ... (main function remains the same)
//     cv::Mat image_mat(480, 640, CV_8UC3, cv::Scalar(255, 255, 255));
//     for (int i = 0; i < image_mat.rows; i += 32) cv::line(image_mat, {0, i}, {image_mat.cols, i}, {0, 0, 0}, 1);
//     for (int i = 0; i < image_mat.cols; i += 32) cv::line(image_mat, {i, 0}, {i, image_mat.rows}, {0, 0, 0}, 1);
//     cv::circle(image_mat, {320, 240}, 100, {255, 0, 0}, -1);
//     torch::Tensor image = xt::utils::image::mat_to_tensor_float(image_mat);
//     xt::transforms::image::RandomResizedCrop transformer({224, 224}, {0.08, 1.0}, {0.75, 1.33}, "bicubic");
//     torch::Tensor transformed_tensor = std::any_cast<torch::Tensor>(transformer.forward({image}));
//     cv::Mat transformed_mat = xt::utils::image::tensor_to_mat_8u(transformed_tensor);
//     cv::imwrite("rrcrop_after.png", transformed_mat);
//     return 0;
// }


namespace xt::transforms::image {

    RandomResizedCrop::RandomResizedCrop() : RandomResizedCrop({224, 224}) {}

    RandomResizedCrop::RandomResizedCrop(
        std::pair<int, int> size,
        std::pair<double, double> scale,
        std::pair<double, double> ratio,
        const std::string& interpolation)
        : size_(size), scale_(scale), ratio_(ratio), interpolation_mode_str_(interpolation) {

        // --- Parameter Validation ---
        if (size_.first <= 0 || size_.second <= 0) {
            throw std::invalid_argument("Output size must be positive.");
        }
        if (scale_.first > scale_.second || ratio_.first > ratio_.second) {
            throw std::invalid_argument("Scale and ratio ranges must be valid (min <= max).");
        }

        if (interpolation_mode_str_ != "bilinear" &&
            interpolation_mode_str_ != "nearest" &&
            interpolation_mode_str_ != "bicubic") {
            throw std::invalid_argument("Unsupported interpolation type.");
        }

        std::random_device rd;
        gen_.seed(rd());
    }

    std::tuple<int, int, int, int> RandomResizedCrop::get_params(
        const torch::Tensor& img,
        const std::pair<double, double>& scale,
        const std::pair<double, double>& ratio,
        std::mt19937& gen)
    {
        // ... (this function remains the same as before)
        auto img_h = img.size(1);
        auto img_w = img.size(2);
        double area = img_h * img_w;
        for (int i = 0; i < 10; ++i) {
            std::uniform_real_distribution<> area_dist(scale.first, scale.second);
            std::uniform_real_distribution<> log_ratio_dist(std::log(ratio.first), std::log(ratio.second));
            double target_area = area * area_dist(gen);
            double aspect_ratio = std::exp(log_ratio_dist(gen));
            int w = static_cast<int>(std::round(std::sqrt(target_area * aspect_ratio)));
            int h = static_cast<int>(std::round(std::sqrt(target_area / aspect_ratio)));
            if (w > 0 && h > 0 && w <= img_w && h <= img_h) {
                std::uniform_int_distribution<int> top_dist(0, img_h - h);
                std::uniform_int_distribution<int> left_dist(0, img_w - w);
                return {top_dist(gen), left_dist(gen), h, w};
            }
        }
        double in_ratio = (double)img_w / (double)img_h;
        int w, h;
        if (in_ratio < ratio.first) {
            w = img_w;
            h = static_cast<int>(std::round(w / ratio.first));
        } else if (in_ratio > ratio.second) {
            h = img_h;
            w = static_cast<int>(std::round(h * ratio.second));
        } else {
            w = img_w;
            h = img_h;
        }
        int top = (img_h - h) / 2;
        int left = (img_w - w) / 2;
        return {top, left, h, w};
    }

    auto RandomResizedCrop::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomResizedCrop::forward received an empty list.");
        }
        torch::Tensor img = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!img.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomResizedCrop is not defined.");
        }

        // --- Step 1: Get Random Crop Parameters ---
        auto [top, left, height, width] = get_params(img, scale_, ratio_, gen_);

        // --- Step 2: Crop the Image ---
        torch::Tensor cropped_img = img.slice(/*dim=*/1, /*start=*/top, /*end=*/top + height)
                                       .slice(/*dim=*/2, /*start=*/left, /*end=*/left + width);

        // --- Step 3: Resize to Final Size ---
        namespace F = torch::nn::functional;

        // ** THE CORRECTED LOGIC IS HERE **
        // Build the interpolation options dynamically based on the stored string.
        auto interp_opts = F::InterpolateFuncOptions()
            .size(std::vector<int64_t>{size_.first, size_.second});

        if (interpolation_mode_str_ == "bilinear") {
            interp_opts.mode(torch::kBilinear).align_corners(false);
        } else if (interpolation_mode_str_ == "bicubic") {
            interp_opts.mode(torch::kBicubic).align_corners(false);
        } else { // "nearest"
            interp_opts.mode(torch::kNearest);
        }

        return F::interpolate(
            cropped_img.unsqueeze(0), // interpolate needs a batch dimension
            interp_opts
        ).squeeze(0); // remove the batch dimension
    }

} // namespace xt::transforms::image