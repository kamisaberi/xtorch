#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::target
{
    class EventToSignalConverter final : public xt::Module
    {
    public:
        EventToSignalConverter();
        explicit EventToSignalConverter(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
