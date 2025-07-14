#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    struct CenterCrop : public xt::Module
    {
    public:
        explicit CenterCrop(std::vector<int64_t> size);


        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::vector<int64_t> size;
    };
}
