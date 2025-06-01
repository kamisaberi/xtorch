#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::target
{
    class EventToIntervalConverter final : public xt::Module
    {
    public:
        EventToIntervalConverter();
        explicit EventToIntervalConverter(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
