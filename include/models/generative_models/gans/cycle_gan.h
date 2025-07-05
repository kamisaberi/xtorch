#pragma once

#include "../../common.h"


namespace xt::models
{
    struct CycleGAN : xt::Cloneable<CycleGAN>
    {
    private:

    public:
        struct ResidualBlock : torch::nn::Module
        {
        public:
            ResidualBlock(int64_t channels);
            torch::Tensor forward(torch::Tensor x);

        private:
            torch::nn::Conv2d conv1, conv2;
            torch::nn::InstanceNorm2d norm1, norm2;
        };


        struct Generator : torch::nn::Module
        {
        public:
            Generator();
            torch::Tensor forward(torch::Tensor x);

        private:
            torch::nn::Conv2d conv_in, conv_out;
            torch::nn::Conv2d conv_down1, conv_down2;
            torch::nn::ConvTranspose2d conv_up1, conv_up2;
            torch::nn::InstanceNorm2d norm_in, norm_down1, norm_down2, norm_up1, norm_up2;
            std::shared_ptr<ResidualBlock> res1, res2, res3, res4, res5, res6;
        };

        struct Discriminator : torch::nn::Module
        {
        public:
            Discriminator();

            torch::Tensor forward(torch::Tensor x);

        private:
            torch::nn::Conv2d conv1, conv2, conv3, conv4;
            torch::nn::InstanceNorm2d norm2, norm3;
        };


        CycleGAN(int num_classes /* classes */, int in_channels = 3/* input channels */);

        CycleGAN(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        void reset() override;
    };
}
