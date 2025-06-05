#include "include/models/computer_vision/object_detection/yolo.h"


using namespace std;

namespace xt::models
{
    YoloV1::YoloV1(int num_classes, int in_channels)
    {
    }

    YoloV1::YoloV1(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void YoloV1::reset()
    {
    }

    auto YoloV1::forward(std::initializer_list<std::any> tensors) -> std::any
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



    YoloV2::YoloV2(int num_classes, int in_channels)
    {
    }

    YoloV2::YoloV2(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void YoloV2::reset()
    {
    }

    auto YoloV2::forward(std::initializer_list<std::any> tensors) -> std::any
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


    YoloV3::YoloV3(int num_classes, int in_channels)
    {
    }

    YoloV3::YoloV3(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void YoloV3::reset()
    {
    }

    auto YoloV3::forward(std::initializer_list<std::any> tensors) -> std::any
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



    YoloV4::YoloV4(int num_classes, int in_channels)
    {
    }

    YoloV4::YoloV4(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void YoloV4::reset()
    {
    }

    auto YoloV4::forward(std::initializer_list<std::any> tensors) -> std::any
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



    YoloV5::YoloV5(int num_classes, int in_channels)
    {
    }

    YoloV5::YoloV5(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void YoloV5::reset()
    {
    }

    auto YoloV5::forward(std::initializer_list<std::any> tensors) -> std::any
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




    YoloV6::YoloV6(int num_classes, int in_channels)
    {
    }

    YoloV6::YoloV6(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void YoloV6::reset()
    {
    }

    auto YoloV6::forward(std::initializer_list<std::any> tensors) -> std::any
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
