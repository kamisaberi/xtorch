#pragma once

#include "../../headers/transforms.h"

namespace xt::transforms::image {


    struct Cutout {
    public:
        Cutout(int num_holes = 1, int hole_size = 16);

        torch::Tensor operator()(const torch::Tensor& input_tensor) const;
    private:
        int num_holes;
        int hole_size;

    };

}