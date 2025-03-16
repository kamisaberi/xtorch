#pragma once
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

using namespace std;


namespace torch::ext::models {
   struct AlexNet :public torch::nn::Module {
        torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr, layer5 = nullptr;
        torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;
    public:
        AlexNet(int num_classes /* classes */, int in_channels =3/* input channels */);

        torch::Tensor forward(torch::Tensor x);
    };
}
