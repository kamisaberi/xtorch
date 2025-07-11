#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <torch/torch.h>
using namespace std;


namespace xt::utils::image
{
    torch::Tensor convertImageToTensor(std::filesystem::path image, vector<int> size = {0, 0});
    torch::Tensor resize(const torch::Tensor& tensor, const std::vector<int64_t>& size);
    cv::Mat tensor_to_mat_local(torch::Tensor tensor);
    torch::Tensor mat_to_tensor_local(const cv::Mat& mat);
    cv::Mat tensor_to_mat_8u(torch::Tensor tensor);
    cv::Mat tensor_to_mat_8u(torch::Tensor tensor);
    torch::Tensor mat_to_tensor_float(const cv::Mat& mat);
}
