#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <torch/torch.h>
using namespace std;

namespace fs = std::filesystem;

namespace torch::ext::media::opencv {

    torch::Tensor convertImageToTensor(fs::path image ,vector<int> size = {0,0});



}


