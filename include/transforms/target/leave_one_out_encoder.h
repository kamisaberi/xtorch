#pragma once

#include "../common.h"


namespace xt::transforms::target
{
    class LeaveOneOutEncoder final : public xt::Module
    {
    public:
        LeaveOneOutEncoder();
        explicit LeaveOneOutEncoder(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
