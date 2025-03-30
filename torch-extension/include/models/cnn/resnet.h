#pragma once
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "../base.h"


using namespace std;

namespace torch::ext::models {

    namespace {
        struct ResidualBlock : BaseModel {
            torch::nn::Sequential conv1 = nullptr, conv2 = nullptr, downsample = nullptr;
            int out_channels;
            torch::nn::ReLU relu = nullptr;
            torch::Tensor residual;

            ResidualBlock(int in_channels, int out_channels, int stride = 1,
                          torch::nn::Sequential downsample = nullptr);

            torch::Tensor forward(torch::Tensor x);
        };
    }

    struct ResNet : BaseModel {
        int inplanes = 64;
        torch::nn::Sequential conv1 = nullptr;
        torch::nn::MaxPool2d maxpool = nullptr;
        torch::nn::AvgPool2d avgpool = nullptr;
        torch::nn::Sequential layer0 = nullptr, layer1 = nullptr, layer2 = nullptr, layer3 = nullptr;
        torch::nn::Linear fc = nullptr;

        ResNet(vector<int> layers, int num_classes = 10, int in_channels = 3 /* input channels */);

        torch::nn::Sequential makeLayerFromResidualBlock(int planes, int blocks, int stride = 1);

        torch::Tensor forward(torch::Tensor x) const override;
    };

}