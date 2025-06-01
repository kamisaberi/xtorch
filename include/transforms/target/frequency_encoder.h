#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::target
{
    class FrequencyEncoder final : public xt::Module
    {
    public:
        FrequencyEncoder();
        explicit FrequencyEncoder(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
