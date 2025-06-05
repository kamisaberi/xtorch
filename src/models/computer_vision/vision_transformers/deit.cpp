#include "include/models/computer_vision/vision_transformers/deit.h"


using namespace std;

namespace xt::models
{
    DeiT::DeiT(int num_classes, int in_channels)
    {
    }

    DeiT::DeiT(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DeiT::reset()
    {
    }

    auto DeiT::forward(std::initializer_list<std::any> tensors) -> std::any
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
