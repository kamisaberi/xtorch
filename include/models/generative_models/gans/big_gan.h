#pragma once

#include "../../common.h"


namespace xt::models
{
    struct ConditionalBatchNorm : public torch::nn::Module
    {
    public:
        ConditionalBatchNorm(int64_t num_features, int64_t num_classes);

        torch::Tensor forward(torch::Tensor x, torch::Tensor y);

    private:
        torch::nn::BatchNorm2d bn;
        torch::nn::Linear gamma_embed, beta_embed;
    };

    struct SpectralNormConv : torch::nn::Module
    {
    public:
        SpectralNormConv(torch::nn::Conv2dOptions options);

        torch::Tensor forward(torch::Tensor x);

    private:
        torch::nn::Conv2d conv;
        torch::Tensor u;
    };


    struct BigGAN : xt::Cloneable<BigGAN>
    {
    public:
        BigGAN(int64_t z_dim, int64_t num_classes);


        struct GeneratorBlock : torch::nn::Module
        {
        public:
            GeneratorBlock(int64_t in_channels, int64_t out_channels, int64_t num_classes);
            torch::Tensor forward(torch::Tensor x, torch::Tensor y);

        private:
            std::shared_ptr<ConditionalBatchNorm> bn1, bn2;
            std::shared_ptr<SpectralNormConv> conv1, conv2;
        };


        torch::nn::Linear linear;
        std::shared_ptr<GeneratorBlock> block1, block2;
        std::shared_ptr<ConditionalBatchNorm> bn;
        std::shared_ptr<SpectralNormConv> conv;


        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        void reset() override;
    };
}
