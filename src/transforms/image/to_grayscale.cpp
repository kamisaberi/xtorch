#include "../../../include/transforms/image/to_grayscale.h"

namespace xt::data::transforms {




    torch::Tensor ToGray::operator()(const torch::Tensor& color_tensor) const {
        // Convert CHW to HWC
        auto img_tensor = color_tensor.detach().cpu().clone().permute({1, 2, 0});
        img_tensor = img_tensor.mul(255).clamp(0, 255).to(torch::kU8);

        // Convert to OpenCV Mat
        cv::Mat img(img_tensor.size(0), img_tensor.size(1), CV_8UC3);
        std::memcpy(img.data, img_tensor.data_ptr(), sizeof(uint8_t) * img_tensor.numel());

        // Convert to grayscale
        cv::Mat gray_img;
        cv::cvtColor(img, gray_img, cv::COLOR_RGB2GRAY);

        // Convert to tensor: [H, W] -> [1, H, W]
        auto tensor = torch::from_blob(
            gray_img.data, {1, gray_img.rows, gray_img.cols}, torch::kUInt8).clone();

        return tensor.to(torch::kFloat32).div(255);  // Normalize to [0, 1]
    }


}