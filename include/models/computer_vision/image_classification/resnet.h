#pragma once
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "../../base.h"


using namespace std;

namespace xt::models {
    namespace {
        struct ResidualBlock : torch::nn::Module {
            torch::nn::Sequential conv1 = nullptr, conv2 = nullptr, downsample = nullptr;
            int out_channels;
            torch::nn::ReLU relu = nullptr;
            torch::Tensor residual;

            ResidualBlock(int in_channels, int out_channels, int stride = 1,
                          torch::nn::Sequential downsample = nullptr);

            torch::Tensor forward(torch::Tensor x);
        };
    }

    struct ResNet18 : BaseModel {
        mutable int inplanes = 64;
        mutable torch::nn::Sequential conv1 = nullptr;
        mutable torch::nn::MaxPool2d maxpool = nullptr;
        mutable torch::nn::AvgPool2d avgpool = nullptr;

        mutable torch::nn::Sequential layer0 = nullptr, layer1 = nullptr, layer2 = nullptr, layer3 = nullptr;
        mutable torch::nn::Linear fc = nullptr;

        ResNet18(vector<int> layers, int num_classes = 10, int in_channels = 3 /* input channels */);

        ResNet18(std::vector<int> layers, int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::nn::Sequential makeLayerFromResidualBlock(int planes, int blocks, int stride = 1);

        torch::Tensor forward(torch::Tensor x) const override;
    };


    struct ResNet34 : BaseModel {
        mutable int inplanes = 64;
        mutable torch::nn::Sequential conv1 = nullptr;
        mutable torch::nn::MaxPool2d maxpool = nullptr;
        mutable torch::nn::AvgPool2d avgpool = nullptr;

        mutable torch::nn::Sequential layer0 = nullptr, layer1 = nullptr, layer2 = nullptr, layer3 = nullptr;
        mutable torch::nn::Linear fc = nullptr;

        ResNet34(vector<int> layers, int num_classes = 10, int in_channels = 3 /* input channels */);

        ResNet34(std::vector<int> layers, int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::nn::Sequential makeLayerFromResidualBlock(int planes, int blocks, int stride = 1);

        torch::Tensor forward(torch::Tensor x) const override;
    };



    struct ResNet50 : BaseModel {
        mutable int inplanes = 64;
        mutable torch::nn::Sequential conv1 = nullptr;
        mutable torch::nn::MaxPool2d maxpool = nullptr;
        mutable torch::nn::AvgPool2d avgpool = nullptr;

        mutable torch::nn::Sequential layer0 = nullptr, layer1 = nullptr, layer2 = nullptr, layer3 = nullptr;
        mutable torch::nn::Linear fc = nullptr;

        ResNet50(vector<int> layers, int num_classes = 10, int in_channels = 3 /* input channels */);

        ResNet50(std::vector<int> layers, int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::nn::Sequential makeLayerFromResidualBlock(int planes, int blocks, int stride = 1);

        torch::Tensor forward(torch::Tensor x) const override;
    };


    struct ResNet101 : BaseModel {
        mutable int inplanes = 64;
        mutable torch::nn::Sequential conv1 = nullptr;
        mutable torch::nn::MaxPool2d maxpool = nullptr;
        mutable torch::nn::AvgPool2d avgpool = nullptr;

        mutable torch::nn::Sequential layer0 = nullptr, layer1 = nullptr, layer2 = nullptr, layer3 = nullptr;
        mutable torch::nn::Linear fc = nullptr;

        ResNet101(vector<int> layers, int num_classes = 10, int in_channels = 3 /* input channels */);

        ResNet101(std::vector<int> layers, int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::nn::Sequential makeLayerFromResidualBlock(int planes, int blocks, int stride = 1);

        torch::Tensor forward(torch::Tensor x) const override;
    };


    struct ResNet152 : BaseModel {
        mutable int inplanes = 64;
        mutable torch::nn::Sequential conv1 = nullptr;
        mutable torch::nn::MaxPool2d maxpool = nullptr;
        mutable torch::nn::AvgPool2d avgpool = nullptr;

        mutable torch::nn::Sequential layer0 = nullptr, layer1 = nullptr, layer2 = nullptr, layer3 = nullptr;
        mutable torch::nn::Linear fc = nullptr;

        ResNet152(vector<int> layers, int num_classes = 10, int in_channels = 3 /* input channels */);

        ResNet152(std::vector<int> layers, int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::nn::Sequential makeLayerFromResidualBlock(int planes, int blocks, int stride = 1);

        torch::Tensor forward(torch::Tensor x) const override;
    };


    struct ResNet200 : BaseModel {
        mutable int inplanes = 64;
        mutable torch::nn::Sequential conv1 = nullptr;
        mutable torch::nn::MaxPool2d maxpool = nullptr;
        mutable torch::nn::AvgPool2d avgpool = nullptr;

        mutable torch::nn::Sequential layer0 = nullptr, layer1 = nullptr, layer2 = nullptr, layer3 = nullptr;
        mutable torch::nn::Linear fc = nullptr;

        ResNet200(vector<int> layers, int num_classes = 10, int in_channels = 3 /* input channels */);

        ResNet200(std::vector<int> layers, int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::nn::Sequential makeLayerFromResidualBlock(int planes, int blocks, int stride = 1);

        torch::Tensor forward(torch::Tensor x) const override;
    };


    struct ResNet1202 : BaseModel {
        mutable int inplanes = 64;
        mutable torch::nn::Sequential conv1 = nullptr;
        mutable torch::nn::MaxPool2d maxpool = nullptr;
        mutable torch::nn::AvgPool2d avgpool = nullptr;

        mutable torch::nn::Sequential layer0 = nullptr, layer1 = nullptr, layer2 = nullptr, layer3 = nullptr;
        mutable torch::nn::Linear fc = nullptr;

        ResNet1202(vector<int> layers, int num_classes = 10, int in_channels = 3 /* input channels */);

        ResNet1202(std::vector<int> layers, int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::nn::Sequential makeLayerFromResidualBlock(int planes, int blocks, int stride = 1);

        torch::Tensor forward(torch::Tensor x) const override;
    };



}
