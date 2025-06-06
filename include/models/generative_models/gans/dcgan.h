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
            Discriminator(int nc, int ndf)
                : conv1(torch::nn::Conv2dOptions(nc, ndf, 4).stride(2).padding(1).bias(false)),
                  conv2(torch::nn::Conv2dOptions(ndf, ndf * 2, 4).stride(2).padding(1).bias(false)),
                  conv3(torch::nn::Conv2dOptions(ndf * 2, ndf * 4, 4).stride(2).padding(1).bias(false)),
                  conv4(torch::nn::Conv2dOptions(ndf * 4, ndf * 8, 4).stride(2).padding(1).bias(false)),
                  conv5(torch::nn::Conv2dOptions(ndf * 8, 1, 4).stride(1).padding(0).bias(false)),
                  bn1(torch::nn::BatchNorm2d(ndf * 2)),
                  bn2(torch::nn::BatchNorm2d(ndf * 4)),
                  bn3(torch::nn::BatchNorm2d(ndf * 8))
            {
                register_module("conv1", conv1);
                register_module("conv2", conv2);
                register_module("conv3", conv3);
                register_module("conv4", conv4);
                register_module("conv5", conv5);
                register_module("bn1", bn1);
                register_module("bn2", bn2);
                register_module("bn3", bn3);

                // Initialize weights
                apply([](torch::nn::Module& module)
                {
                    if (auto* conv = dynamic_cast<torch::nn::Conv2dImpl*>(&module))
                    {
                        torch::nn::init::normal_(conv->weight, 0.0, 0.02);
                    }
                    else if (auto* bn = dynamic_cast<torch::nn::BatchNorm2dImpl*>(&module))
                    {
                        torch::nn::init::normal_(bn->weight, 1.0, 0.02);
                        torch::nn::init::constant_(bn->bias, 0.0);
                    }
                });
            }

            torch::Tensor forward(torch::Tensor x)
            {
                x = torch::leaky_relu(conv1(x), 0.2);
                x = torch::leaky_relu(bn1(conv2(x)), 0.2);
                x = torch::leaky_relu(bn2(conv3(x)), 0.2);
                x = torch::leaky_relu(bn3(conv4(x)), 0.2);
                x = torch::sigmoid(conv5(x));
                return x;
            }

            torch::nn::Conv2d conv1, conv2, conv3, conv4, conv5;
            torch::nn::BatchNorm2d bn1, bn2, bn3;
        };
    };
}
