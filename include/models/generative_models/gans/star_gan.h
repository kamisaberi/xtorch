#pragma once
#include "../../common.h"


namespace xt::models
{
    struct StarGAN : xt::Cloneable<StarGAN>
    {
    private:

    public:
        // // Generator
        struct StarGANGeneratorImpl : torch::nn::Module
        {
            StarGANGeneratorImpl(int num_domains);

            torch::Tensor forward(torch::Tensor x, torch::Tensor domain_label);

            int num_domains_;
            torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
            torch::nn::ConvTranspose2d deconv3{nullptr}, deconv2{nullptr}, deconv1{nullptr};
            torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr}, bn5{nullptr};
            torch::nn::ReLU relu{nullptr};
            torch::nn::Tanh tanh{nullptr};
        };

        TORCH_MODULE(StarGANGenerator);

        // Discriminator with Auxiliary Classifier
        struct StarGANDiscriminatorImpl : torch::nn::Module
        {
            StarGANDiscriminatorImpl(int num_domains);

            std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

            int num_domains_;
            torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv_src{nullptr}, conv_cls{nullptr};
            torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
            torch::nn::LeakyReLU lrelu{nullptr};
        };

        TORCH_MODULE(StarGANDiscriminator);


        StarGAN(int num_classes /* classes */, int in_channels = 3/* input channels */);

        StarGAN(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        void reset() override;
    };
}
