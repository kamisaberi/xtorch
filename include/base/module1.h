#pragma once
#include <any>
#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>
#include <string>
#include <vector>


namespace xt
{
    class Module1: public torch::nn::Module {
    public:
        Module1();
        // Pure virtual forward with initializer_list
        virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;

        // Operator() to call forward
        auto operator()(std::initializer_list<std::any> tensors) -> std::any;

        // Pure virtual destructor
        ~Module1() override = default;    };
}