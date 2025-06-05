#include "include/models/natural_language_processing/transformers/reformer.h"


using namespace std;

namespace xt::models
{
    Reformer::Reformer(int num_classes, int in_channels)
    {
    }

    Reformer::Reformer(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void Reformer::reset()
    {
    }

    auto Reformer::forward(std::initializer_list<std::any> tensors) -> std::any
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
