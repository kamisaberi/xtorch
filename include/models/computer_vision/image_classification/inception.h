#pragma once

#include "models/common.h"


using namespace std;


namespace xt::models
{
    struct InceptionV1 : xt::Cloneable<InceptionV1>
    {
    private:

    public:
        InceptionV1(int num_classes /* classes */, int in_channels = 3/* input channels */);

        InceptionV1(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

    struct InceptionV2 : xt::Cloneable<InceptionV2>
    {
    private:

    public:
        InceptionV2(int num_classes /* classes */, int in_channels = 3/* input channels */);

        InceptionV2(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };



}
