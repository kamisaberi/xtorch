#pragma once
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

using namespace std;


namespace torch::ext::cnn::models {
   struct AlexNet : torch::nn::Module {
        torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr, layer5 = nullptr;
        torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;
    public:
        AlexNet(int num_classes);

        torch::Tensor forward(torch::Tensor x);
    };
}
