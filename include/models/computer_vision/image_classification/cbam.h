#pragma once

#include "../../common.h"


using namespace std;

namespace xt::models
{
    struct CBAM : xt::Cloneable<CBAM>
    {
    protected:

    public:
        explicit CBAM(int num_classes/* classes */, int in_channels = 1/*  input channels */);
        CBAM(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };
}
