#pragma once
#include "../../headers/transforms.h"

namespace xt::data::transforms {



    struct RandomGaussianBlur {
    private:
        std::vector<int> kernel_sizes; // List of odd kernel sizes to choose from
        double sigma_min;
        double sigma_max;

    public:
        RandomGaussianBlur(std::vector<int> sizes = {3, 5}, double sigma_min = 0.1, double sigma_max = 2.0);

        torch::Tensor operator()(const torch::Tensor &input_tensor);
    };



}