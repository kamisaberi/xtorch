//TODO SHOULD CHANGE
#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    struct CenterCrop : public xt::Module
    {
    public:
        explicit CenterCrop(std::vector<int64_t> size);

        torch::Tensor operator()(torch::Tensor input);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::vector<int64_t> size;
    };
}
