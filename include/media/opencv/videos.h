#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <torch/torch.h>
#include <string>
#include <vector>

using namespace std;

namespace fs = std::filesystem;

namespace torch::ext::media::opencv::videos {
    std::vector<cv::Mat>  extractFrames(const std::string& videoFilePath);
    std::vector<torch::Tensor> extractVideoFramesAsTensor(fs::path videoFilePath);
}
