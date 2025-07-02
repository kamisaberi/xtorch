//TODO SHOULD CHANGE
#pragma once
#include "../common.h"


namespace xt::transforms::image
{
    struct RandomFlip
    {
    private:
        double horizontal_prob;
        double vertical_prob;

    public:
        RandomFlip(double h_prob = 0.5, double v_prob = 0.0);

        torch::Tensor operator()(const torch::Tensor& input_tensor);
    };
}
