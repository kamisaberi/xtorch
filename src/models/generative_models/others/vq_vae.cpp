#include "include/models/generative_models/others/vq_vae.h"


using namespace std;

namespace xt::models
{
    QAVAEV1::QAVAEV1(int num_classes, int in_channels)
    {
    }

    QAVAEV1::QAVAEV1(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void QAVAEV1::reset()
    {
    }

    auto QAVAEV1::forward(std::initializer_list<std::any> tensors) -> std::any
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
