#pragma once

#include "../../common.h"


namespace xt::models
{
    struct StyleGAN : xt::Cloneable<StyleGAN>
    {
    private:

    public:
        // Adaptive Instance Normalization (AdaIN)
        torch::Tensor ada_in(torch::Tensor x, torch::Tensor ys, torch::Tensor yb)
        {
            auto mean = x.mean({2, 3}, true);
            auto std = x.std({2, 3}, true);
            auto normalized = (x - mean) / (std + 1e-5);
            auto ys_expanded = ys.unsqueeze(2).unsqueeze(3);
            auto yb_expanded = yb.unsqueeze(2).unsqueeze(3);
            return ys_expanded * normalized + yb_expanded;
        }

        // Mapping Network: 8-layer MLP (z -> w)
        struct MappingNetwork : torch::nn::Module
        {
            MappingNetwork(int64_t z_dim, int64_t w_dim)
                : fc0(z_dim, w_dim), fc1(w_dim, w_dim), fc2(w_dim, w_dim), fc3(w_dim, w_dim),
                  fc4(w_dim, w_dim), fc5(w_dim, w_dim), fc6(w_dim, w_dim), fc7(w_dim, w_dim)
            {
                register_module("fc0", fc0);
                register_module("fc1", fc1);
                register_module("fc2", fc2);
                register_module("fc3", fc3);
                register_module("fc4", fc4);
                register_module("fc5", fc5);
                register_module("fc6", fc6);
                register_module("fc7", fc7);
            }

            torch::Tensor forward(torch::Tensor z)
            {
                auto x = torch::leaky_relu(fc0->forward(z), 0.2);
                x = torch::leaky_relu(fc1->forward(x), 0.2);
                x = torch::leaky_relu(fc2->forward(x), 0.2);
                x = torch::leaky_relu(fc3->forward(x), 0.2);
                x = torch::leaky_relu(fc4->forward(x), 0.2);
                x = torch::leaky_relu(fc5->forward(x), 0.2);
                x = torch::leaky_relu(fc6->forward(x), 0.2);
                x = torch::leaky_relu(fc7->forward(x), 0.2);
                return x;
            }

            torch::nn::Linear fc0, fc1, fc2, fc3, fc4, fc5, fc6, fc7;
        };

        // Synthesis Network Block
        struct SynthesisBlock : torch::nn::Module
        {
            SynthesisBlock(int64_t in_channels, int64_t out_channels, int64_t w_dim)
                : conv(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)),
                  style_s(torch::nn::Linear(w_dim, out_channels)),
                  style_b(torch::nn::Linear(w_dim, out_channels))
            {
                register_module("conv", conv);
                register_module("style_s", style_s);
                register_module("style_b", style_b);
            }

            torch::Tensor forward(torch::Tensor x, torch::Tensor w)
            {
                x = conv->forward(x);
                auto ys = style_s->forward(w);
                auto yb = style_b->forward(w);
                x = ada_in(x, ys, yb);
                return torch::leaky_relu(x, 0.2);
            }

            torch::nn::Conv2d conv;
            torch::nn::Linear style_s, style_b;
        };

        // Synthesis Network
        struct SynthesisNetwork : torch::nn::Module
        {
            SynthesisNetwork(int64_t w_dim)
                : initial(torch::nn::Linear(w_dim, 4 * 4 * 64)),
                  block1(64, 64, w_dim), block2(64, 32, w_dim), block3(32, 16, w_dim),
                  to_rgb(torch::nn::Conv2dOptions(16, 1, 1))
            {
                register_module("initial", initial);
                register_module("block1", block1);
                register_module("block2", block2);
                register_module("block3", block3);
                register_module("to_rgb", to_rgb);
            }

            torch::Tensor forward(torch::Tensor w)
            {
                auto x = initial->forward(w).view({-1, 64, 4, 4});
                x = block1->forward(x, w);
                x = torch::upsample_nearest2d(x, {8, 8});
                x = block2->forward(x, w);
                x = torch::upsample_nearest2d(x, {16, 16});
                x = block3->forward(x, w);
                x = torch::upsample_nearest2d(x, {64, 64});
                x = to_rgb->forward(x);
                return torch::tanh(x);
            }

            torch::nn::Linear initial;
            SynthesisBlock block1, block2, block3;
            torch::nn::Conv2d to_rgb;
        };

        // StyleGAN Generator
        struct StyleGANGenerator : torch::nn::Module
        {
            StyleGANGenerator(int64_t z_dim, int64_t w_dim)
                : mapping(z_dim, w_dim), synthesis(w_dim)
            {
                register_module("mapping", mapping);
                register_module("synthesis", synthesis);
            }

            torch::Tensor forward(torch::Tensor z)
            {
                auto w = mapping->forward(z);
                return synthesis->forward(w);
            }

            MappingNetwork mapping;
            SynthesisNetwork synthesis;
        };


        StyleGAN(int num_classes /* classes */, int in_channels = 3/* input channels */);

        StyleGAN(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        void reset() override;
    };
}
