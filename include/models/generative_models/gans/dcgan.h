#pragma once
#include "../../common.h"


namespace xt::models
{
    // Generator Network


    struct DCGAN : xt::Cloneable<DCGAN>
    {
    private:

    public:
        DCGAN(int num_classes /* classes */, int in_channels = 3/* input channels */);
        DCGAN(int num_classes, int in_channels, std::vector<int64_t> input_shape);
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        void reset() override;


        struct Generator : torch::nn::Module
        {
        public:
            Generator(int nz, int ngf, int nc);
            torch::Tensor forward(torch::Tensor x);

        private:
            torch::nn::ConvTranspose2d conv1, conv2, conv3, conv4, conv5;
            torch::nn::BatchNorm2d bn1, bn2, bn3, bn4;
        };


        // Discriminator Network
        struct Discriminator : torch::nn::Module
        {
        public:
            Discriminator(int nc, int ndf);
            torch::Tensor forward(torch::Tensor x);

        private:
            torch::nn::Conv2d conv1, conv2, conv3, conv4, conv5;
            torch::nn::BatchNorm2d bn1, bn2, bn3;
        };
    };
}
