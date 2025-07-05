#pragma once

#include "../../common.h"


namespace xt::models
{
    struct ConditionalBatchNorm : public torch::nn::Module
    {
        ConditionalBatchNorm(int64_t num_features, int64_t num_classes)
            : bn(num_features),
              gamma_embed(torch::nn::Linear(num_classes, num_features)),
              beta_embed(torch::nn::Linear(num_classes, num_features))
        {
            register_module("bn", bn);
            register_module("gamma_embed", gamma_embed);
            register_module("beta_embed", beta_embed);
        }

        torch::Tensor forward(torch::Tensor x, torch::Tensor y)
        {
            auto out = bn->forward(x);
            auto gamma = gamma_embed->forward(y).unsqueeze(2).unsqueeze(3);
            auto beta = beta_embed->forward(y).unsqueeze(2).unsqueeze(3);
            return gamma * out + beta;
        }

        torch::nn::BatchNorm2d bn;
        torch::nn::Linear gamma_embed, beta_embed;
    };

    struct SpectralNormConv : xt::Module
    {
        SpectralNormConv(torch::nn::Conv2dOptions options)
            : conv(options)
        {
            register_module("conv", conv);
            u = register_buffer("u", torch::randn({1, options.out_channels()}).normal_(0, 1));
        }

        torch::Tensor forward(torch::Tensor x)
        {
            // Power iteration for spectral norm
            auto weight = conv->weight.view({conv->weight.size(0), -1});
            auto v = torch::matmul(weight.t(), u.t()).t();
            v = v / (v.norm() + 1e-8);
            auto sigma = torch::matmul(u, torch::matmul(weight, v)).item<float>();
            return conv->forward(x) / (sigma + 1e-8);
        }

        torch::nn::Conv2d conv;
        torch::Tensor u;
    };


    struct BigGAN : xt::Cloneable<BigGAN>
    {
    public:
        BigGAN(int num_classes /* classes */, int in_channels = 3/* input channels */);
        BigGAN(int num_classes, int in_channels, std::vector<int64_t> input_shape);


        struct GeneratorBlock : torch::nn::Module
        {
            GeneratorBlock(int64_t in_channels, int64_t out_channels, int64_t num_classes)
            {
                bn1 = std::make_shared<ConditionalBatchNorm>(in_channels, num_classes);
                register_module("bn1", bn1);
                conv1 = std::make_shared<SpectralNormConv>(
                    torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1));
                register_module("conv1", conv1);
                bn2 = std::make_shared<ConditionalBatchNorm>(out_channels, num_classes);
                register_module("bn2", bn2);
                conv2 = std::make_shared<SpectralNormConv>(
                    torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1));
                register_module("conv2", conv2);
            }

            torch::Tensor forward(torch::Tensor x, torch::Tensor y)
            {
                auto out = torch::relu(bn1->forward(x, y));
                out = torch::upsample_nearest2d(out, 2);
                out = conv1->forward(out);
                out = torch::relu(bn2->forward(out, y));
                out = conv2->forward(out);
                return out;
            }

            std::shared_ptr<ConditionalBatchNorm> bn1, bn2;
            std::shared_ptr<SpectralNormConv> conv1, conv2;
        };

        // BigGAN Generator
        struct BigGANGenerator : torch::nn::Module
        {
            BigGANGenerator(int64_t z_dim, int64_t num_classes)
                : linear(torch::nn::Linear(z_dim + num_classes, 4 * 4 * 256))

            {
                register_module("linear", linear);
                block1 = std::make_shared<GeneratorBlock>(256, 128, num_classes);
                register_module("block1", block1);
                block2 = std::make_shared<GeneratorBlock>(128, 64, num_classes);
                register_module("block2", block2);
                bn = std::make_shared<ConditionalBatchNorm>(64, num_classes);
                register_module("bn", bn);
                conv = std::make_shared<SpectralNormConv>(torch::nn::Conv2dOptions(64, 1, 3).padding(1));
                register_module("conv", conv);
            }

            torch::Tensor forward(torch::Tensor z, torch::Tensor y)
            {
                auto input = torch::cat({z, y}, 1);
                auto out = linear->forward(input).view({-1, 256, 4, 4});
                out = block1->forward(out, y);
                out = block2->forward(out, y);
                out = torch::relu(bn->forward(out, y));
                out = torch::tanh(conv->forward(out));
                return out;
            }

            torch::nn::Linear linear;
            std::shared_ptr<GeneratorBlock> block1, block2;
            std::shared_ptr<ConditionalBatchNorm> bn;
            std::shared_ptr<SpectralNormConv> conv;
        };


        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        void reset() override;
    };
}
