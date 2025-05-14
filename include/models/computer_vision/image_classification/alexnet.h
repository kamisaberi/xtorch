#pragma once

#include "models/common.h"

using namespace std;


namespace xt::models {
    struct AlexNet final : xt::Module {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr, layer5 =
                        nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

    public:
        explicit AlexNet(int num_classes /* classes */, int in_channels = 3/* input channels */);

        AlexNet(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
    };
}
