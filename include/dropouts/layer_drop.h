#pragma once

#include "common.h"

namespace xt::dropouts {

    struct LayerDrop : xt::Module {
    public:
        LayerDrop(double p_drop_layer = 0.1) ;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        double p_drop_layer_; // Probability of dropping the layer this module gates.

    };
}



