#pragma once
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "../../base.h"


using namespace std;


namespace xt::models {
    struct EfficientNetB0 : BaseModel {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr, layer5 =
                nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

    public:
        EfficientNetB0(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB0(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
    };

    struct EfficientNetB1 : BaseModel {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr, layer5 =
                nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

    public:
        EfficientNetB1(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB1(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
    };

    struct EfficientNetB2 : BaseModel {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr, layer5 =
                nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

    public:
        EfficientNetB2(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB2(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
    };
    struct EfficientNetB3 : BaseModel {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr, layer5 =
                nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

    public:
        EfficientNetB3(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB3(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
    };

    struct EfficientNetB4 : BaseModel {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr, layer5 =
                nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

    public:
        EfficientNetB4(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB4(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
    };
    struct EfficientNetB5 : BaseModel {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr, layer5 =
                nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

    public:
        EfficientNetB5(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB5(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
    };

    struct EfficientNetB6 : BaseModel {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr, layer5 =
                nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

    public:
        EfficientNetB6(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB6(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
    };


    struct EfficientNetB7 : BaseModel {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr, layer5 =
                nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

    public:
        EfficientNetB7(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB7(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
    };


}
