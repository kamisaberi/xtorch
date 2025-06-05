#pragma once

#include "../../common.h"


using namespace std;


namespace xt::models
{

    struct InceptionResNetV1 : xt::Cloneable<InceptionResNetV1>
    {
    private:

    public:
        InceptionResNetV1(int num_classes /* classes */, int in_channels = 3/* input channels */);

        InceptionResNetV1(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

        void reset() override;
    };

    struct InceptionResNetV2 : xt::Cloneable<InceptionResNetV2>
    {
    private:

    public:
        InceptionResNetV2(int num_classes /* classes */, int in_channels = 3/* input channels */);

        InceptionResNetV2(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

        void reset() override;
    };


}
