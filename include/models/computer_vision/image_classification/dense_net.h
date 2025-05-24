#pragma once
#include "include/models/common.h"


using namespace std;


namespace xt::models
{
    struct DenseNet121 : xt::Cloneable<DenseNet121>
    {
    private:

    public:
        DenseNet121(int num_classes /* classes */, int in_channels = 3/* input channels */);

        DenseNet121(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };


    struct DenseNet169 : xt::Cloneable<DenseNet169>
    {
    private:

    public:
        DenseNet169(int num_classes /* classes */, int in_channels = 3/* input channels */);

        DenseNet169(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };


    struct DenseNet201 : xt::Cloneable<DenseNet201>
    {
    private:

    public:
        DenseNet201(int num_classes /* classes */, int in_channels = 3/* input channels */);

        DenseNet201(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };


    struct DenseNet264 : xt::Cloneable<DenseNet264>
    {
    private:

    public:
        DenseNet264(int num_classes /* classes */, int in_channels = 3/* input channels */);

        DenseNet264(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };
}
