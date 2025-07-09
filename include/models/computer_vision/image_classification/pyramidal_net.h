#pragma once

#include "../../common.h"


namespace xt::models {
    // --- The Core Pyramidal Residual Block ---

    struct PyramidalBasicBlockImpl : torch::nn::Module {
        torch::nn::Conv2d conv1{nullptr}, conv2_nobn{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};

        int stride = 1;

        PyramidalBasicBlockImpl(int in_planes, int out_planes, int stride = 1)
                : stride(stride) {
            conv1 = register_module("conv1", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(in_planes, out_planes, 3).stride(stride).padding(1).bias(false)));
            bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_planes));

            // The second conv maps back to the same number of output planes
            conv2_nobn = register_module("conv2", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(out_planes, out_planes, 3).stride(1).padding(1).bias(false)));
            bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_planes));
        }

        torch::Tensor forward(torch::Tensor x) {
            auto out = torch::relu(bn1(conv1(x)));
            out = bn2(conv2_nobn(out));

            // --- The Pyramidal Shortcut Connection ---
            // This is the key part of the architecture.
            // If dimensions change, we use a zero-padding shortcut.
            torch::Tensor shortcut = x;
            int in_channels = x.size(1);
            int out_channels = out.size(1);

            if (stride != 1 || in_channels != out_channels) {
                // Downsample spatially if stride is not 1
                if (stride != 1) {
                    shortcut = torch::nn::functional::avg_pool2d(shortcut,
                                                                 torch::nn::functional::AvgPool2dFuncOptions(2));
                }

                // Pad with zero-channels to match output dimensions
                int64_t pad_channels = out_channels - in_channels;
                if (pad_channels > 0) {
                    auto padding = torch::zeros({shortcut.size(0), pad_channels, shortcut.size(2), shortcut.size(3)},
                                                shortcut.options());
                    shortcut = torch::cat({shortcut, padding}, 1);
                }
            }

            out += shortcut;
            out = torch::relu(out);
            return out;
        }
    };

    TORCH_MODULE(PyramidalBasicBlock);


    // --- The Full PyramidalNet Model ---

    struct PyramidalNetImpl : torch::nn::Module {
        torch::nn::Conv2d conv1;
        torch::nn::BatchNorm2d bn1;
        torch::nn::Sequential layer1, layer2, layer3;
        torch::nn::Linear linear;

        PyramidalNetImpl(int N, int alpha, int num_classes = 10) {
            // For MNIST, we use a simple stem
            conv1 = register_module("conv1", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(1, 16, 3).stride(1).padding(1).bias(false)));
            bn1 = register_module("bn1", torch::nn::BatchNorm2d(16));

            // Calculate how many channels to add at each block
            const double add_per_block = static_cast<double>(alpha) / N;
            double current_planes = 16.0;

            // Stage 1
            layer1 = torch::nn::Sequential();
            for (int i = 0; i < N; ++i) {
                int in_planes = static_cast<int>(round(current_planes));
                current_planes += add_per_block;
                int out_planes = static_cast<int>(round(current_planes));
                layer1->push_back(PyramidalBasicBlock(in_planes, out_planes, 1));
            }
            register_module("layer1", layer1);

            // Stage 2
            layer2 = torch::nn::Sequential();
            for (int i = 0; i < N; ++i) {
                int stride = (i == 0) ? 2 : 1; // Downsample at the start of the stage
                int in_planes = static_cast<int>(round(current_planes));
                current_planes += add_per_block;
                int out_planes = static_cast<int>(round(current_planes));
                layer2->push_back(PyramidalBasicBlock(in_planes, out_planes, stride));
            }
            register_module("layer2", layer2);

            // Stage 3
            layer3 = torch::nn::Sequential();
            for (int i = 0; i < N; ++i) {
                int stride = (i == 0) ? 2 : 1;
                int in_planes = static_cast<int>(round(current_planes));
                current_planes += add_per_block;
                int out_planes = static_cast<int>(round(current_planes));
                layer3->push_back(PyramidalBasicBlock(in_planes, out_planes, stride));
            }
            register_module("layer3", layer3);

            // Final classifier
            int final_planes = static_cast<int>(round(current_planes));
            linear = register_module("linear", torch::nn::Linear(final_planes, num_classes));
        }

        torch::Tensor forward(torch::Tensor x) {
            x = torch::relu(bn1(conv1(x)));
            x = layer1->forward(x);
            x = layer2->forward(x);
            x = layer3->forward(x);

            // Global Average Pooling
            x = torch::nn::functional::adaptive_avg_pool2d(x,
                                                           torch::nn::functional::AdaptiveAvgPool2dFuncOptions(1));

            x = x.view({x.size(0), -1});
            x = linear->forward(x);
            return x;
        }
    };

    TORCH_MODULE(PyramidalNet);

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
