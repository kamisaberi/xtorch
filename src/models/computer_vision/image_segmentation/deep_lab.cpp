#include "include/models/computer_vision/image_segmentation/deep_lab.h"


using namespace std;

namespace xt::models
{
    DeepLabV1::DeepLabV1(int num_classes, int in_channels)
    {
    }

    DeepLabV1::DeepLabV1(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DeepLabV1::reset()
    {
    }

    auto DeepLabV1::forward(std::initializer_list<std::any> tensors) -> std::any
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


    DeepLabV2::DeepLabV2(int num_classes, int in_channels)
    {
    }

    DeepLabV2::DeepLabV2(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DeepLabV2::reset()
    {
    }

    auto DeepLabV2::forward(std::initializer_list<std::any> tensors) -> std::any
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






    DeepLabV3::DeepLabV3(int num_classes, int in_channels)
    {
    }

    DeepLabV3::DeepLabV3(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DeepLabV3::reset()
    {
    }

    auto DeepLabV3::forward(std::initializer_list<std::any> tensors) -> std::any
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



    DeepLabV3Plus::DeepLabV3Plus(int num_classes, int in_channels)
    {
    }

    DeepLabV3Plus::DeepLabV3Plus(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DeepLabV3Plus::reset()
    {
    }

    auto DeepLabV3Plus::forward(std::initializer_list<std::any> tensors) -> std::any
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
