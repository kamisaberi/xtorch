#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::image
{
    struct CenterCrop {
    public:
        explicit CenterCrop(std::vector<int64_t> size);

        torch::Tensor operator()(torch::Tensor input);

    private:
        std::vector<int64_t> size;
    };

}