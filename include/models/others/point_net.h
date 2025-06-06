#pragma once
#include "../common.h"

namespace xt::models
{
    struct PointNet : xt::Cloneable<PointNet>
    {
    private:

    public:
        PointNet(int num_classes /* classes */, int in_channels = 3/* input channels */);

        PointNet(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };

    struct PointNetPlusPlus : xt::Cloneable<PointNetPlusPlus>
    {
    private:

    public:
        PointNetPlusPlus(int num_classes /* classes */, int in_channels = 3/* input channels */);

        PointNetPlusPlus(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };

}