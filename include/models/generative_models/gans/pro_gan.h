#pragma once
#include "../../common.h"


namespace xt::models
{
    struct ProGAN : xt::Cloneable<ProGAN>
    {
    private:

    public:
        struct GeneratorBlockImpl : torch::nn::Module
        {
            GeneratorBlockImpl(int in_channels, int out_channels);

            torch::Tensor forward(torch::Tensor x);

            torch::nn::ConvTranspose2d conv{nullptr};
            torch::nn::BatchNorm2d bn{nullptr};
            torch::nn::ReLU relu{nullptr};
        };

        TORCH_MODULE(GeneratorBlock);


        struct ProGANGeneratorImpl : torch::nn::Module
        {
            ProGANGeneratorImpl(int latent_dim);

            torch::Tensor forward(torch::Tensor z, int resolution_level, float alpha);

            int latent_dim_;
            torch::nn::Linear initial{nullptr};
            GeneratorBlock block1{nullptr}, block2{nullptr}, block3{nullptr};
            torch::nn::Conv2d to_img_4x4{nullptr}, to_img_8x8{nullptr}, to_img_16x16{nullptr}, to_img_28x28{nullptr};
            torch::nn::ReLU relu{nullptr};
        };

        TORCH_MODULE(ProGANGenerator);

        // Discriminator Block
        struct DiscriminatorBlockImpl : torch::nn::Module
        {
            DiscriminatorBlockImpl(int in_channels, int out_channels);

            torch::Tensor forward(torch::Tensor x);

            torch::nn::Conv2d conv{nullptr};
            torch::nn::BatchNorm2d bn{nullptr};
            torch::nn::LeakyReLU lrelu{nullptr};
        };

        TORCH_MODULE(DiscriminatorBlock);

        // Discriminator
        struct ProGANDiscriminatorImpl : torch::nn::Module
        {
            ProGANDiscriminatorImpl();

            torch::Tensor forward(torch::Tensor x, int resolution_level, float alpha);

            torch::nn::Conv2d from_img_28x28{nullptr}, from_img_16x16{nullptr}, from_img_8x8{nullptr}, from_img_4x4{
                                  nullptr
                              };
            DiscriminatorBlock block3{nullptr}, block2{nullptr}, block1{nullptr};
            torch::nn::Linear final{nullptr};
            torch::nn::LeakyReLU lrelu{nullptr};
        };

        TORCH_MODULE(ProGANDiscriminator);


        ProGAN(int num_classes /* classes */, int in_channels = 3/* input channels */);

        ProGAN(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        void reset() override;
    };
}
