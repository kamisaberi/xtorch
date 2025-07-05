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
            ProGANGeneratorImpl(int latent_dim) : latent_dim_(latent_dim)
            {
                initial = register_module("initial", torch::nn::Linear(latent_dim, 256 * 4 * 4));
                block1 = register_module("block1", GeneratorBlock(256, 128)); // 4x4 -> 8x8
                block2 = register_module("block2", GeneratorBlock(128, 64)); // 8x8 -> 16x16
                block3 = register_module("block3", GeneratorBlock(64, 32)); // 16x16 -> 28x28
                to_img_4x4 = register_module("to_img_4x4", torch::nn::Conv2d(
                                                 torch::nn::Conv2dOptions(256, 1, 1).stride(1)));
                to_img_8x8 = register_module("to_img_8x8", torch::nn::Conv2d(
                                                 torch::nn::Conv2dOptions(128, 1, 1).stride(1)));
                to_img_16x16 = register_module("to_img_16x16", torch::nn::Conv2d(
                                                   torch::nn::Conv2dOptions(64, 1, 1).stride(1)));
                to_img_28x28 = register_module("to_img_28x28", torch::nn::Conv2d(
                                                   torch::nn::Conv2dOptions(32, 1, 3).stride(1).padding(1)));
                relu = register_module("relu", torch::nn::ReLU());
            }

            torch::Tensor forward(torch::Tensor z, int resolution_level, float alpha)
            {
                auto batch_size = z.size(0);
                z = relu->forward(initial->forward(z)); // [batch, 256 * 4 * 4]
                z = z.view({batch_size, 256, 4, 4}); // [batch, 256, 4, 4]

                if (resolution_level == 0)
                {
                    return torch::tanh(to_img_4x4->forward(z)); // [batch, 1, 4, 4]
                }

                z = block1->forward(z); // [batch, 128, 8, 8]
                if (resolution_level == 1)
                {
                    auto img = to_img_8x8->forward(z);
                    return torch::tanh(img);
                }
                else if (resolution_level == 2)
                {
                    auto img_prev = torch::upsample_nearest2d(to_img_4x4->forward(block1->forward(z)), {8, 8});
                    auto img_curr = to_img_8x8->forward(z);
                    return torch::tanh((1 - alpha) * img_prev + alpha * img_curr); // Fade-in
                }

                z = block2->forward(z); // [batch, 64, 16, 16]
                if (resolution_level == 3)
                {
                    auto img = to_img_16x16->forward(z);
                    return torch::tanh(img);
                }
                else if (resolution_level == 4)
                {
                    auto img_prev = torch::upsample_nearest2d(to_img_8x8->forward(block2->forward(z)), {16, 16});
                    auto img_curr = to_img_16x16->forward(z);
                    return torch::tanh((1 - alpha) * img_prev + alpha * img_curr); // Fade-in
                }

                z = block3->forward(z); // [batch, 32, 28, 28]
                if (resolution_level == 5)
                {
                    auto img = to_img_28x28->forward(z);
                    return torch::tanh(img);
                }
                else if (resolution_level == 6)
                {
                    auto img_prev = torch::upsample_nearest2d(to_img_16x16->forward(block2->forward(z)), {28, 28});
                    auto img_curr = to_img_28x28->forward(z);
                    return torch::tanh((1 - alpha) * img_prev + alpha * img_curr); // Fade-in
                }

                throw std::runtime_error("Invalid resolution level");
            }

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
            DiscriminatorBlockImpl(int in_channels, int out_channels)
            {
                conv = register_module("conv", torch::nn::Conv2d(
                                           torch::nn::Conv2dOptions(in_channels, out_channels,
                                                                    4).stride(2).padding(1)));
                bn = register_module("bn", torch::nn::BatchNorm2d(out_channels));
                lrelu = register_module("lrelu", torch::nn::LeakyReLU(
                                            torch::nn::LeakyReLUOptions().negative_slope(0.2)));
            }

            torch::Tensor forward(torch::Tensor x)
            {
                x = lrelu->forward(bn->forward(conv->forward(x)));
                return x;
            }

            torch::nn::Conv2d conv{nullptr};
            torch::nn::BatchNorm2d bn{nullptr};
            torch::nn::LeakyReLU lrelu{nullptr};
        };

        TORCH_MODULE(DiscriminatorBlock);

        // Discriminator
        struct ProGANDiscriminatorImpl : torch::nn::Module
        {
            ProGANDiscriminatorImpl()
            {
                from_img_28x28 = register_module("from_img_28x28", torch::nn::Conv2d(
                                                     torch::nn::Conv2dOptions(1, 32, 3).stride(1).padding(1)));
                from_img_16x16 = register_module("from_img_16x16", torch::nn::Conv2d(
                                                     torch::nn::Conv2dOptions(1, 64, 1).stride(1)));
                from_img_8x8 = register_module("from_img_8x8", torch::nn::Conv2d(
                                                   torch::nn::Conv2dOptions(1, 128, 1).stride(1)));
                from_img_4x4 = register_module("from_img_4x4", torch::nn::Conv2d(
                                                   torch::nn::Conv2dOptions(1, 256, 1).stride(1)));
                block3 = register_module("block3", DiscriminatorBlock(32, 64)); // 28x28 -> 16x16
                block2 = register_module("block2", DiscriminatorBlock(64, 128)); // 16x16 -> 8x8
                block1 = register_module("block1", DiscriminatorBlock(128, 256)); // 8x8 -> 4x4
                final = register_module("final", torch::nn::Linear(256 * 4 * 4, 1));
                lrelu = register_module("lrelu", torch::nn::LeakyReLU(
                                            torch::nn::LeakyReLUOptions().negative_slope(0.2)));
            }

            torch::Tensor forward(torch::Tensor x, int resolution_level, float alpha)
            {
                auto batch_size = x.size(0);

                if (resolution_level == 0)
                {
                    x = lrelu->forward(from_img_4x4->forward(x)); // [batch, 256, 4, 4]
                }
                else if (resolution_level == 1 || resolution_level == 2)
                {
                    x = lrelu->forward(from_img_8x8->forward(x)); // [batch, 128, 8, 8]
                    if (resolution_level == 2)
                    {
                        auto x_prev = lrelu->forward(from_img_4x4->forward(torch::avg_pool2d(x, 2)));
                        x = (1 - alpha) * x_prev + alpha * x; // Fade-in
                    }
                    x = block1->forward(x); // [batch, 256, 4, 4]
                }
                else if (resolution_level == 3 || resolution_level == 4)
                {
                    x = lrelu->forward(from_img_16x16->forward(x)); // [batch, 64, 16, 16]
                    if (resolution_level == 4)
                    {
                        auto x_prev = lrelu->forward(from_img_8x8->forward(torch::avg_pool2d(x, 2)));
                        x = (1 - alpha) * x_prev + alpha * x; // Fade-in
                    }
                    x = block2->forward(x); // [batch, 128, 8, 8]
                    x = block1->forward(x); // [batch, 256, 4, 4]
                }
                else if (resolution_level == 5 || resolution_level == 6)
                {
                    x = lrelu->forward(from_img_28x28->forward(x)); // [batch, 32, 28, 28]
                    if (resolution_level == 6)
                    {
                        auto x_prev = lrelu->forward(from_img_16x16->forward(torch::avg_pool2d(x, 2)));
                        x = (1 - alpha) * x_prev + alpha * x; // Fade-in
                    }
                    x = block3->forward(x); // [batch, 64, 16, 16]
                    x = block2->forward(x); // [batch, 128, 8, 8]
                    x = block1->forward(x); // [batch, 256, 4, 4]
                }
                else
                {
                    throw std::runtime_error("Invalid resolution level");
                }

                x = x.view({batch_size, -1}); // [batch, 256 * 4 * 4]
                x = final->forward(x); // [batch, 1]
                return x;
            }

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
