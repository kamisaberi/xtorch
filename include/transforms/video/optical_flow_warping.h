#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::video
{
    class OpticalFlowWarping final : public xt::Module
    {
    public:
        OpticalFlowWarping();
        explicit OpticalFlowWarping(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
