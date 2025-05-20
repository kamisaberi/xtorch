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


    struct InceptionV2 : BaseModel
    {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr, layer5 =
                                          nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

    public:
        InceptionV2(int num_classes /* classes */, int in_channels = 3/* input channels */);

        InceptionV2(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
    };

    struct InceptionV3 : BaseModel
    {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr, layer5 =
                                          nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

    public:
        InceptionV3(int num_classes /* classes */, int in_channels = 3/* input channels */);

        InceptionV3(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
    };

    struct InceptionV4 : BaseModel
    {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr, layer5 =
                                          nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

    public:
        InceptionV4(int num_classes /* classes */, int in_channels = 3/* input channels */);

        InceptionV4(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
    };
}
