#include "include/models/natural_language_processing/others/word2vec.h"


using namespace std;

namespace xt::models
{
    Word2Vec::Word2Vec(int num_classes, int in_channels)
    {
    }

    Word2Vec::Word2Vec(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void Word2Vec::reset()
    {
    }

    auto Word2Vec::forward(std::initializer_list<std::any> tensors) -> std::any
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
