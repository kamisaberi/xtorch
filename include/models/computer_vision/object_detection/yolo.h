#pragma once
#include "../../common.h"

namespace xt::models
{
    struct YoloV1 : xt::Cloneable<YoloV1>
    {
    private:

    public:
        YoloV1(int num_classes /* classes */, int in_channels = 3/* input channels */);

        YoloV1(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };


    struct YoloV2 : xt::Cloneable<YoloV2>
    {
    private:

    public:
        YoloV2(int num_classes /* classes */, int in_channels = 3/* input channels */);

        YoloV2(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };


    struct YoloV3 : xt::Cloneable<YoloV3>
    {
    private:

    public:
        YoloV3(int num_classes /* classes */, int in_channels = 3/* input channels */);

        YoloV3(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };


    struct YoloV4 : xt::Cloneable<YoloV4>
    {
    private:

    public:
        YoloV4(int num_classes /* classes */, int in_channels = 3/* input channels */);

        YoloV4(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };

    struct YoloV5 : xt::Cloneable<YoloV5>
    {
    private:

    public:
        YoloV5(int num_classes /* classes */, int in_channels = 3/* input channels */);

        YoloV5(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };

    struct YoloV6 : xt::Cloneable<YoloV6>
    {
    private:

    public:
        YoloV6(int num_classes /* classes */, int in_channels = 3/* input channels */);

        YoloV6(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };



}