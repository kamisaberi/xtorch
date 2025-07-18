#include <transforms/image/color_jitter.h>

// --- Example Main (for testing) ---
// #include "transforms/image/color_jitter.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a colorful sample image.
//     cv::Mat image_mat(256, 384, CV_8UC3);
//     // Create three colored bands
//     image_mat(cv::Rect(0, 0, 128, 256)).setTo(cv::Scalar(255, 0, 0));       // Blue
//     image_mat(cv::Rect(128, 0, 128, 256)).setTo(cv::Scalar(0, 255, 0));    // Green
//     image_mat(cv::Rect(256, 0, 128, 256)).setTo(cv::Scalar(0, 0, 255));    // Red
//
//     cv::imwrite("jitter_before.png", image_mat);
//     std::cout << "Saved jitter_before.png" << std::endl;
//
//     torch::Tensor image = xt::utils::image::mat_to_tensor_float(image_mat);
//
//     std::cout << "--- Applying ColorJitter ---" << std::endl;
//
//     // 2. Define transform with strong jitter settings.
//     xt::transforms::image::ColorJitter jitterer(
//         /*brightness=*/std::make_pair(0.5, 1.5),
//         /*contrast=*/std::make_pair(0.5, 1.5),
//         /*saturation=*/std::make_pair(0.5, 1.5),
//         /*hue=*/0.2
//     );
//
//     // 3. Apply the transform
//     torch::Tensor jittered_tensor = std::any_cast<torch::Tensor>(jitterer.forward({image}));
//
//     // 4. Save the result
//     cv::Mat jittered_mat = xt::utils::image::tensor_to_mat_8u(jittered_tensor);
//     cv::imwrite("jitter_after.png", jittered_mat);
//     std::cout << "Saved jitter_after.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    ColorJitter::ColorJitter() {
        // Common default values
        brightness_ = std::make_pair(0.6, 1.4); // 40% jitter
        contrast_   = std::make_pair(0.6, 1.4);
        saturation_ = std::make_pair(0.6, 1.4);
        hue_        = 0.1; // 10% hue jitter
        std::random_device rd;
        gen_.seed(rd());
    }

    ColorJitter::ColorJitter(
        std::optional<std::pair<double, double>> brightness,
        std::optional<std::pair<double, double>> contrast,
        std::optional<std::pair<double, double>> saturation,
        std::optional<double> hue)
        : brightness_(brightness), contrast_(contrast), saturation_(saturation), hue_(hue) {

        // --- Parameter Validation ---
        if (hue_.has_value() && (hue_.value() < 0.0 || hue_.value() > 0.5)) {
            throw std::invalid_argument("Hue jitter must be between 0.0 and 0.5.");
        }
        std::random_device rd;
        gen_.seed(rd());
    }

    auto ColorJitter::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("ColorJitter::forward received an empty list.");
        }
        torch::Tensor img = std::any_cast<torch::Tensor>(any_vec[0]);
        if (!img.defined()) {
            throw std::invalid_argument("Input tensor passed to ColorJitter is not defined.");
        }

        // --- Create a list of transform functions to apply ---
        std::vector<std::function<torch::Tensor(torch::Tensor)>> transforms;

        if (brightness_.has_value()) {
            transforms.emplace_back([this](torch::Tensor in_img) {
                std::uniform_real_distribution<> dist(brightness_->first, brightness_->second);
                double factor = dist(gen_);
                return torch::lerp(torch::zeros_like(in_img), in_img, factor).clamp_(0.0, 1.0);
            });
        }
        if (contrast_.has_value()) {
            transforms.emplace_back([this](torch::Tensor in_img) {
                std::uniform_real_distribution<> dist(contrast_->first, contrast_->second);
                double factor = dist(gen_);
                torch::Tensor mean_gray = torch::mean(in_img, /*dim=*/0, /*keepdim=*/true);
                return torch::lerp(mean_gray, in_img, factor).clamp_(0.0, 1.0);
            });
        }
        if (saturation_.has_value()) {
            transforms.emplace_back([this](torch::Tensor in_img) {
                std::uniform_real_distribution<> dist(saturation_->first, saturation_->second);
                double factor = dist(gen_);
                torch::Tensor weights = torch::tensor({0.299, 0.587, 0.114}, in_img.options()).view({3, 1, 1});
                torch::Tensor grayscale = (in_img * weights).sum(0, true).repeat({3, 1, 1});
                return torch::lerp(grayscale, in_img, factor).clamp_(0.0, 1.0);
            });
        }
        if (hue_.has_value()) {
            transforms.emplace_back([this](torch::Tensor in_img) {
                std::uniform_real_distribution<> dist(-hue_.value(), hue_.value());
                double factor = dist(gen_);
                if (std::abs(factor) < 1e-6) return in_img;

                cv::Mat mat_8u = xt::utils::image::tensor_to_mat_8u(in_img);
                cv::Mat hsv_mat;
                cv::cvtColor(mat_8u, hsv_mat, cv::COLOR_RGB2HSV);

                std::vector<cv::Mat> channels;
                cv::split(hsv_mat, channels);

                // OpenCV Hue is in [0, 179]. A factor of 1.0 is 360 degrees.
                int hue_shift = static_cast<int>(factor * 180.0);
                channels[0] += hue_shift; // wrap around is handled by uchar overflow

                cv::merge(channels, hsv_mat);
                cv::cvtColor(hsv_mat, mat_8u, cv::COLOR_HSV2RGB);

                return xt::utils::image::mat_to_tensor_float(mat_8u);
            });
        }

        // --- Shuffle the order and apply the transforms ---
        std::shuffle(transforms.begin(), transforms.end(), gen_);

        for (const auto& transform_fn : transforms) {
            img = transform_fn(img);
        }

        return img;
    }

} // namespace xt::transforms::image