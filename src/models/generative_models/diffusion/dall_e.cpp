#include "include/models/generative_models/diffusion/dall_e.h"


using namespace std;

namespace xt::models
{
    DallEV1::DallEV1(int num_classes, int in_channels)
    {
    }

    DallEV1::DallEV1(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DallEV1::reset()
    {
    }

    auto DallEV1::forward(std::initializer_list<std::any> tensors) -> std::any
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

    DallEV2::DallEV2(int num_classes, int in_channels)
    {
    }

    DallEV2::DallEV2(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DallEV2::reset()
    {
    }

    auto DallEV2::forward(std::initializer_list<std::any> tensors) -> std::any
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

    DallEV3::DallEV3(int num_classes, int in_channels)
    {
    }

    DallEV3::DallEV3(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DallEV3::reset()
    {
    }

    auto DallEV3::forward(std::initializer_list<std::any> tensors) -> std::any
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
