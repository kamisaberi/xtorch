#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <torch/torch.h>
using namespace std;

namespace fs = std::filesystem;

namespace xt::utils::image {


    torch::Tensor convertImageToTensor(fs::path image ,vector<int> size = {0,0});
    torch::Tensor resize(const torch::Tensor &tensor, const std::vector<int64_t> &size);

}


