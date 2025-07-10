#pragma once

#include "../../common.h"


namespace xt::models {
    // --- The Core Pyramidal Residual Block ---

    // struct PyramidalBasicBlock : xt::Module {
    //     torch::nn::Conv2d conv1{nullptr}, conv2_nobn{nullptr};
    //     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    //
    //     int stride = 1;
    //
    //     PyramidalBasicBlock(int in_planes, int out_planes, int stride = 1);
    //     auto forward(std::initializer_list <std::any> tensors) -> std::any override;
    //
    //     torch::Tensor forward(torch::Tensor x);
    // };
    //
    // // TORCH_MODULE(PyramidalBasicBlock);
    //
    //
    // // --- The Full PyramidalNet Model ---
    //
    // struct PyramidalNet : xt::Module {
    //     torch::nn::Conv2d conv1;
    //     torch::nn::BatchNorm2d bn1;
    //     torch::nn::Sequential layer1, layer2, layer3;
    //     torch::nn::Linear linear;
    //
    //     PyramidalNet(int N, int alpha, int num_classes = 10);
    //     auto forward(std::initializer_list <std::any> tensors) -> std::any override;
    //
    //     torch::Tensor forward(torch::Tensor x);
    // };

    // TORCH_MODULE(PyramidalNet);

//    struct PyramidalNet : xt::Cloneable<PyramidalNet>
//    {
//    private:
//
//    public:
//        PyramidalNet(int num_classes /* classes */, int in_channels = 3/* input channels */);
//
//        PyramidalNet(int num_classes, int in_channels, std::vector<int64_t> input_shape);
//
//        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
//
//        void reset() override;
//    };
}
