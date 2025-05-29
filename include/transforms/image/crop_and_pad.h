#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class CropAndPad final : public xt::Module
    {
    public:
        CropAndPad();
        explicit CropAndPad(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
