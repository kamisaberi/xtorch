#include "../../include/transforms/image/color_jitter.h"

namespace xt::data::transforms {



    ColorJitter::ColorJitter(float brightness_, float contrast_, float saturation_)
        : brightness(brightness_), contrast(contrast_), saturation(saturation_) {}

    torch::Tensor ColorJitter::operator()(const torch::Tensor& input_tensor) const {
        static thread_local std::mt19937 gen(std::random_device{}());

        // Convert CHW -> HWC, [0,1] -> [0,255]
        auto img_tensor = input_tensor.detach().cpu().clone()
            .permute({1, 2, 0})
            .mul(255).clamp(0, 255)
            .to(torch::kU8);

        cv::Mat img(img_tensor.size(0), img_tensor.size(1), CV_8UC3);
        std::memcpy(img.data, img_tensor.data_ptr(), sizeof(uint8_t) * img_tensor.numel());

        // Convert to float32 OpenCV image
        img.convertTo(img, CV_32F, 1.0 / 255.0);

        // Apply brightness
        if (brightness > 0.0f) {
            std::uniform_real_distribution<> b_dist(1.0 - brightness, 1.0 + brightness);
            img *= b_dist(gen);
        }

        // Apply contrast
        if (contrast > 0.0f) {
            std::uniform_real_distribution<> c_dist(1.0 - contrast, 1.0 + contrast);
            cv::Mat mean;
            cv::cvtColor(img, mean, cv::COLOR_RGB2GRAY);
            cv::Scalar m = cv::mean(mean);
            img = (img - m[0]) * c_dist(gen) + m[0];
        }

        // Apply saturation
        if (saturation > 0.0f) {
            std::uniform_real_distribution<> s_dist(1.0 - saturation, 1.0 + saturation);
            float s_factor = s_dist(gen);
            cv::Mat gray;
            cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
            cv::cvtColor(gray, gray, cv::COLOR_GRAY2RGB);
            img = img * s_factor + gray * (1.0 - s_factor);
        }

        // Convert back to torch tensor: [H, W, C] -> [C, H, W], [0,1]
        torch::Tensor output = torch::from_blob(
            img.data, {img.rows, img.cols, 3}, torch::kFloat32).clone();

        return output.permute({2, 0, 1}).clamp(0, 1);
    }

}