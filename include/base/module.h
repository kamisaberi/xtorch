#pragma once
#include <any>
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
        // Pure virtual forward with initializer_list
        virtual auto forward(std::initializer_list<torch::Tensor> tensors) -> std::any = 0;

        // Operator() to call forward
        auto operator()(std::initializer_list<torch::Tensor> tensors) -> std::any;

        // Pure virtual destructor
        ~Module() override = default;    };
}