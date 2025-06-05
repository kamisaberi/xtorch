#include "include/models/computer_vision/image_classification/inception_resnet.h"


using namespace std;

namespace xt::models
{
    InceptionResNetV1::InceptionResNetV1(int num_classes, int in_channels)
    {
    }

    InceptionResNetV1::InceptionResNetV1(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void InceptionResNetV1::reset()
    {
    }

    auto InceptionResNetV1::forward(std::initializer_list<std::any> tensors) -> std::any
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


    InceptionResNetV2::InceptionResNetV2(int num_classes, int in_channels)
    {
    }

    InceptionResNetV2::InceptionResNetV2(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void InceptionResNetV2::reset()
    {
    }

    auto InceptionResNetV2::forward(std::initializer_list<std::any> tensors) -> std::any
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
