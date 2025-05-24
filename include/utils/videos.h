#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <torch/torch.h>
#include <string>
#include <vector>



namespace xt::utils::videos {
    std::vector<cv::Mat>  extractFrames(const std::string& videoFilePath);
    std::vector<torch::Tensor> extractVideoFramesAsTensor(const std::filesystem::path& videoFilePath);
}
