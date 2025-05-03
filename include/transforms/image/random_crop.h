#pragma once
#include "../../headers/transforms.h"

namespace xt::transforms::image
{
    struct RandomCrop {
    public:
        RandomCrop(std::vector<int64_t> size);

        torch::Tensor operator()(torch::Tensor input);

    private:
        std::vector<int64_t> size;
    };


    struct RandomCrop2 {
    private:
        int crop_height;
        int crop_width;

    public:
        RandomCrop2(int height, int width);

        torch::Tensor operator()(const torch::Tensor &input_tensor);
    };


}