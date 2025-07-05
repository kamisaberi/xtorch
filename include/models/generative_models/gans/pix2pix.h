#pragma once
#include "../../common.h"


namespace xt::models
{
    struct Pix2Pix : xt::Cloneable<Pix2Pix>
    {
    private:

    public:
        // U-Net Generator
        struct UNetGeneratorImpl : torch::nn::Module
        {
            UNetGeneratorImpl();

            torch::Tensor forward(torch::Tensor x);

            torch::nn::Conv2d enc1{nullptr}, enc2{nullptr}, enc3{nullptr};
            torch::nn::ConvTranspose2d dec3{nullptr}, dec2{nullptr}, dec1{nullptr};
            torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr}, bn5{nullptr};
            torch::nn::ReLU relu{nullptr};
            torch::nn::Tanh tanh{nullptr};
        };

        TORCH_MODULE(UNetGenerator);

        // PatchGAN Discriminator
        struct PatchGANDiscriminatorImpl : torch::nn::Module
        {
            PatchGANDiscriminatorImpl();

            torch::Tensor forward(torch::Tensor input, torch::Tensor target);

            torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
            torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
            torch::nn::LeakyReLU lrelu{nullptr};
        };

        TORCH_MODULE(PatchGANDiscriminator);


        Pix2Pix(int num_classes /* classes */, int in_channels = 3/* input channels */);

        Pix2Pix(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        void reset() override;
    };
}
