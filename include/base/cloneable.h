#pragma once
#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>
#include <string>
#include <vector>


namespace xt
{
    class CloneableModule: public torch::nn::Cloneable<CloneableModule> {
    public:
        CloneableModule();
        virtual  torch::Tensor forward(torch::Tensor input) const = 0;
        torch::Tensor operator()(torch::Tensor input) const ;
        void reset() override;


    };
}