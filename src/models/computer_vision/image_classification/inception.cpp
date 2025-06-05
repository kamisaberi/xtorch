#include "include/models/computer_vision/image_classification/inception.h"


using namespace std;

namespace xt::models
{
    InceptionV1::InceptionV1(int num_classes, int in_channels)
    {
    }

    InceptionV1::InceptionV1(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void InceptionV1::reset()
    {
    }

    auto InceptionV1::forward(std::initializer_list<std::any> tensors) -> std::any
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





    InceptionV2::InceptionV2(int num_classes, int in_channels)
    {
    }

    InceptionV2::InceptionV2(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void InceptionV2::reset()
    {
    }

    auto InceptionV2::forward(std::initializer_list<std::any> tensors) -> std::any
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


    InceptionV3::InceptionV3(int num_classes, int in_channels)
    {
    }

    InceptionV3::InceptionV3(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void InceptionV3::reset()
    {
    }

    auto InceptionV3::forward(std::initializer_list<std::any> tensors) -> std::any
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


    InceptionV4::InceptionV4(int num_classes, int in_channels)
    {
    }

    InceptionV4::InceptionV4(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void InceptionV4::reset()
    {
    }

    auto InceptionV4::forward(std::initializer_list<std::any> tensors) -> std::any
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
