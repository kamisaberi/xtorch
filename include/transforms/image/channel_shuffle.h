#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class ChannelShuffle final : public xt::Module
    {
    public:
        ChannelShuffle();
        explicit ChannelShuffle(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
