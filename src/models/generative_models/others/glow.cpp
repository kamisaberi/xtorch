#include "include/models/generative_models/others/glow.h"


using namespace std;

namespace xt::models
{
    Glow::Glow(int num_classes, int in_channels)
    {
    }

    Glow::Glow(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void Glow::reset()
    {
    }

    auto Glow::forward(std::initializer_list<std::any> tensors) -> std::any
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
