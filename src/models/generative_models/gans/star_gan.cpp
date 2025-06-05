#include "include/models/generative_models/gans/star_gan.h"


using namespace std;

namespace xt::models
{
    StarGAN::StarGAN(int num_classes, int in_channels)
    {
    }

    StarGAN::StarGAN(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void StarGAN::reset()
    {
    }

    auto StarGAN::forward(std::initializer_list<std::any> tensors) -> std::any
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
