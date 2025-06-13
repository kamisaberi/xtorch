#pragma once

#include "common.h"

namespace xt::dropouts {

    struct DropConnect : xt::Module {
    public:
        explicit DropConnect(double p_drop = 0.5) ;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        double p_drop_;
        double epsilon_ = 1e-7;

    };
}



