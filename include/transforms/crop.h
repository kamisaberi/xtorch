#pragma once

#include "../headers/transforms.h"

namespace xt::data::transforms {


    struct CenterCrop {
    public:
        CenterCrop(std::vector<int64_t> size);

        torch::Tensor operator()(torch::Tensor input);

    private:
        std::vector<int64_t> size;
    };


    struct RandomCrop {
    public:
        RandomCrop(std::vector<int64_t> size);

        torch::Tensor operator()(torch::Tensor input);

    private:
        std::vector<int64_t> size;
    };





}