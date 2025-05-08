#pragma once
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "../../base.h"


using namespace std;


namespace xt::models {
    struct DenseNet121 : BaseModel {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr, layer5 =
                nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

    public:
        DenseNet121(int num_classes /* classes */, int in_channels = 3/* input channels */);

        DenseNet121(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
    };



    struct DenseNet169 : BaseModel {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr, layer5 =
                nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

    public:
        DenseNet169(int num_classes /* classes */, int in_channels = 3/* input channels */);

        DenseNet169(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
    };




    struct DenseNet201 : BaseModel {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr, layer5 =
                nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

    public:
        DenseNet201(int num_classes /* classes */, int in_channels = 3/* input channels */);

        DenseNet201(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
    };



    struct DenseNet264 : BaseModel {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr, layer5 =
                nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

    public:
        DenseNet264(int num_classes /* classes */, int in_channels = 3/* input channels */);

        DenseNet264(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
    };




}
