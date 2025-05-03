#pragma once
#include "../../../include/transforms/general/to_tensor.h"

namespace xt::transforms
{

    torch::Tensor ToTensor::operator()(const cv::Mat &image) const {
        cv::Mat img;

        // Convert grayscale to 3 channels if needed
        if (image.channels() == 1) {
            cv::cvtColor(image, img, cv::COLOR_GRAY2RGB);
        } else if (image.channels() == 4) {
            cv::cvtColor(image, img, cv::COLOR_BGRA2RGB);
        } else {
            cv::cvtColor(image, img, cv::COLOR_BGR2RGB); // Assume BGR
        }

        // Convert uint8 -> float32 and normalize to [0, 1]
        img.convertTo(img, CV_32F, 1.0 / 255.0);

        // Create tensor from OpenCV Mat
        auto tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kFloat32).clone();

        // HWC -> CHW
        tensor = tensor.permute({2, 0, 1});
        return tensor;
    }



}