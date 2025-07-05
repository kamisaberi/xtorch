#pragma once

#include "../../common.h"


namespace xt::models
{
    struct StyleGAN : xt::Cloneable<StyleGAN>
    {
    private:

    public:
        // Adaptive Instance Normalization (AdaIN)
        static torch::Tensor ada_in(torch::Tensor x, torch::Tensor ys, torch::Tensor yb);

        // Mapping Network: 8-layer MLP (z -> w)
        struct MappingNetwork : torch::nn::Module
        {
            MappingNetwork(int64_t z_dim, int64_t w_dim);

            torch::Tensor forward(torch::Tensor z);

            torch::nn::Linear fc0, fc1, fc2, fc3, fc4, fc5, fc6, fc7;
        };

        // Synthesis Network Block
        struct SynthesisBlock : torch::nn::Module
        {
            SynthesisBlock(int64_t in_channels, int64_t out_channels, int64_t w_dim);

            torch::Tensor forward(torch::Tensor x, torch::Tensor w);

            torch::nn::Conv2d conv;
            torch::nn::Linear style_s, style_b;
        };

        // Synthesis Network
        struct SynthesisNetwork : torch::nn::Module
        {
            SynthesisNetwork(int64_t w_dim);

            torch::Tensor forward(torch::Tensor w);

            torch::nn::Linear initial;
            SynthesisBlock block1, block2, block3;
            torch::nn::Conv2d to_rgb;
        };

        // StyleGAN Generator
        struct StyleGANGenerator : torch::nn::Module
        {
            StyleGANGenerator(int64_t z_dim, int64_t w_dim);

            torch::Tensor forward(torch::Tensor z);

            MappingNetwork mapping;
            SynthesisNetwork synthesis;
        };


        StyleGAN(int num_classes /* classes */, int in_channels = 3/* input channels */);

        StyleGAN(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        void reset() override;
    };
}
