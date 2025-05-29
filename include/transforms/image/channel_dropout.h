#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class ChannelDropout final : public xt::Module
    {
    public:
        ChannelDropout();
        explicit ChannelDropout(std::vector<xt::Module> transforms);
        auto ChannelDropout(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
