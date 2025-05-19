#pragma once
#include "models/common.h"


using namespace std;

namespace xt::models
{
    struct VggNet11 : xt::Cloneable<VggNet11>
    {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr;
        mutable torch::nn::Sequential layer5 = nullptr;
        mutable torch::nn::Sequential layer6 = nullptr, layer7 = nullptr, layer8 = nullptr, layer9 = nullptr;
        mutable torch::nn::Sequential layer10 = nullptr;
        mutable torch::nn::Sequential layer11 = nullptr, layer12 = nullptr, layer13 = nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

        VggNet11(int num_classes /* classes */, int in_channels /* input channels */);
        VggNet11(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

    struct VggNet13 : xt::Cloneable<VggNet13>
    {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr;
        mutable torch::nn::Sequential layer5 = nullptr;
        mutable torch::nn::Sequential layer6 = nullptr, layer7 = nullptr, layer8 = nullptr, layer9 = nullptr;
        mutable torch::nn::Sequential layer10 = nullptr;
        mutable torch::nn::Sequential layer11 = nullptr, layer12 = nullptr, layer13 = nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

        VggNet13(int num_classes /* classes */, int in_channels /* input channels */);
        VggNet13(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };


    struct VggNet16 : xt::Cloneable<VggNet16>
    {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr;
        mutable torch::nn::Sequential layer5 = nullptr;
        mutable torch::nn::Sequential layer6 = nullptr, layer7 = nullptr, layer8 = nullptr, layer9 = nullptr;
        mutable torch::nn::Sequential layer10 = nullptr;
        mutable torch::nn::Sequential layer11 = nullptr, layer12 = nullptr, layer13 = nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

        VggNet16(int num_classes /* classes */, int in_channels /* input channels */);
        VggNet16(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };


    struct VggNet19 : xt::Cloneable<VggNet19>
    {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr;
        mutable torch::nn::Sequential layer5 = nullptr;
        mutable torch::nn::Sequential layer6 = nullptr, layer7 = nullptr, layer8 = nullptr, layer9 = nullptr;
        mutable torch::nn::Sequential layer10 = nullptr;
        mutable torch::nn::Sequential layer11 = nullptr, layer12 = nullptr, layer13 = nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

        VggNet19(int num_classes /* classes */, int in_channels /* input channels */);
        VggNet19(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };
}
