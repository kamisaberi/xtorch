//TODO SHOULD CHANGE
#pragma once
#include "include/transforms/common.h"

namespace xt::transforms::image
{
    struct Resize
    {
    public:
        explicit Resize(std::vector<int64_t> size);

        torch::Tensor operator()(torch::Tensor img);

    private:
        std::vector<int64_t> size;
    };
}
