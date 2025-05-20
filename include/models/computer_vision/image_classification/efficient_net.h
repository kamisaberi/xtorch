#pragma once

#include "models/common.h"


using namespace std;


namespace xt::models
{
    struct EfficientNetB0 : xt::Cloneable<EfficientNetB0>
    {
    private:

    public:
        EfficientNetB0(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB0(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };


    struct EfficientNetB1 : xt::Cloneable<EfficientNetB1>
    {
    private:

    public:
        EfficientNetB1(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB1(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };


    struct EfficientNetB2 : xt::Cloneable<EfficientNetB2>
    {
    private:

    public:
        EfficientNetB2(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB2(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

    struct EfficientNetB3 : xt::Cloneable<EfficientNetB3>
    {
    private:

    public:
        EfficientNetB3(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB3(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

    struct EfficientNetB4 : xt::Cloneable<EfficientNetB4>
    {
    private:

    public:
        EfficientNetB4(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB4(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };


}
