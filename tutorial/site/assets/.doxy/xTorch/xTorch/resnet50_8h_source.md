

# File resnet50.h

[**File List**](files.md) **>** [**cnn**](dir_40be95ab8912b8deac694fbe2f8f2654.md) **>** [**resnet**](dir_43ab8c30072399f09a02fdd1f785b21c.md) **>** [**resnet50.h**](resnet50_8h.md)

[Go to the documentation of this file](resnet50_8h.md)


```C++
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

    struct ResNet : BaseModel {
        mutable int inplanes = 64;
        mutable torch::nn::Sequential conv1 = nullptr;
        mutable torch::nn::MaxPool2d maxpool = nullptr;
        mutable torch::nn::AvgPool2d avgpool = nullptr;

        mutable torch::nn::Sequential layer0 = nullptr, layer1 = nullptr, layer2 = nullptr, layer3 = nullptr;
        mutable torch::nn::Linear fc = nullptr;

        ResNet(vector<int> layers, int num_classes = 10, int in_channels = 3 /* input channels */);

        ResNet(std::vector<int> layers, int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::nn::Sequential makeLayerFromResidualBlock(int planes, int blocks, int stride = 1);

        torch::Tensor forward(torch::Tensor x) const override;
    };
}
```


