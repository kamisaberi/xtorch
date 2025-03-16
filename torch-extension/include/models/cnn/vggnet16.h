#pragma once
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

using namespace std;
namespace torch::ext::models {


    struct VggNet16 : torch::nn::Module {
        torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr, layer5 = nullptr;
        torch::nn::Sequential layer6 = nullptr, layer7 = nullptr, layer8 = nullptr, layer9 = nullptr, layer10 = nullptr;
        torch::nn::Sequential layer11 = nullptr, layer12 = nullptr, layer13 = nullptr;
        torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

        VggNet16(int num_classes /* classes */, int in_channels = 3 /* input channels */);

        torch::Tensor forward(torch::Tensor x);
    };
}
