#include "include/models/computer_vision/vision_transformers/swin_transformer.h"


using namespace std;

namespace xt::models
{
    SwinTransformerV1::SwinTransformerV1(int num_classes, int in_channels)
    {
    }

    SwinTransformerV1::SwinTransformerV1(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void SwinTransformerV1::reset()
    {
    }

    auto SwinTransformerV1::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];

        return x;
    }

    SwinTransformerV2::SwinTransformerV2(int num_classes, int in_channels)
    {
    }

    SwinTransformerV2::SwinTransformerV2(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void SwinTransformerV2::reset()
    {
    }

    auto SwinTransformerV2::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];

        return x;
    }

}
