#pragma once
#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>
#include <string>
#include <vector>


namespace xt
{
    class Module: public torch::nn::Module {
    public:
        Module();
        virtual  torch::Tensor forward(torch::Tensor input) const = 0;
    };
}