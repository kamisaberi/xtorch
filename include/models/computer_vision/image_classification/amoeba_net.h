#pragma once

#include "../../common.h"


using namespace std;

namespace xt::models
{
    struct AmoabaNet : xt::Cloneable<AmoabaNet>
    {
    protected:

    public:
        explicit AmoabaNet(int num_classes/* classes */, int in_channels = 1/*  input channels */);
        AmoabaNet(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

        void reset() override;
    };
}
