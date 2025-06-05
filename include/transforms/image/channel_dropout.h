#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class ChannelDropout final : public xt::Module
    {
    public:
        ChannelDropout();
        explicit ChannelDropout(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
