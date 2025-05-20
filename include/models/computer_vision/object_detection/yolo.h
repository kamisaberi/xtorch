#pragma once
#include "models/common.h"
namespace xt::models
{
    struct YoloV1 : xt::Cloneable<YoloV1>
    {
    private:

    public:
        YoloV1(int num_classes /* classes */, int in_channels = 3/* input channels */);

        YoloV1(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };


    struct YoloV2 : xt::Cloneable<YoloV2>
    {
    private:

    public:
        YoloV2(int num_classes /* classes */, int in_channels = 3/* input channels */);

        YoloV2(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };


    struct YoloV3 : xt::Cloneable<YoloV3>
    {
    private:

    public:
        YoloV3(int num_classes /* classes */, int in_channels = 3/* input channels */);

        YoloV3(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };




}