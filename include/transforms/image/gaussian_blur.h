#pragma once
#include "../../headers/transforms.h"

namespace xt::transforms::image {


    struct GaussianBlur {
    public:
        GaussianBlur(std::vector<int64_t> kernel_size, float sigma);

        torch::Tensor operator()(torch::Tensor input);

    private:
        std::vector<int64_t> kernel_size;
        float sigma;

        torch::Tensor generate_gaussian_kernel(int64_t k_h, int64_t k_w, float sigma, torch::Device device);
    };


    struct GaussianBlurOpenCV {
    public:
        GaussianBlurOpenCV(int ksize, double sigma_val);

        torch::Tensor operator()(const torch::Tensor &input_tensor);

    private:
        cv::Size kernel_size;
        double sigma;
    };




}