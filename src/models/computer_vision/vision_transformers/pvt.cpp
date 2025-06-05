#include "include/models/computer_vision/vision_transformers/pvt.h"


using namespace std;

namespace xt::models
{
    PVTV1::PVTV1(int num_classes, int in_channels)
    {
    }

    PVTV1::PVTV1(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void PVTV1::reset()
    {
    }

    auto PVTV1::forward(std::initializer_list<std::any> tensors) -> std::any
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



    PVTV2::PVTV2(int num_classes, int in_channels)
    {
    }

    PVTV2::PVTV2(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void PVTV2::reset()
    {
    }

    auto PVTV2::forward(std::initializer_list<std::any> tensors) -> std::any
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
