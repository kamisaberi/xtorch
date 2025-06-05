#include "include/models/natural_language_processing/transformers/t5.h"


using namespace std;

namespace xt::models
{
    T5::T5(int num_classes, int in_channels)
    {
    }

    T5::T5(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void T5::reset()
    {
    }

    auto T5::forward(std::initializer_list<std::any> tensors) -> std::any
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
