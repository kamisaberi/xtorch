#pragma once

#include "../../common.h"


namespace xt::models
{
    struct GAN : xt::Cloneable<GAN>
    {
    private:

    public:
        GAN(int num_classes /* classes */, int in_channels = 3/* input channels */);

        GAN(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        void reset() override;

        struct Generator : torch::nn::Module
        {
        public:
            Generator(int nz, int nhidden, int nout);

            torch::Tensor forward(torch::Tensor x);

        private:
            torch::nn::Linear fc1, fc2, fc3;
        };

        // Discriminator Network
        struct Discriminator : torch::nn::Module
        {
        public:
            Discriminator(int nin, int nhidden);

            torch::Tensor forward(torch::Tensor x);

        private:
            torch::nn::Linear fc1, fc2, fc3;
        };
    };
}
