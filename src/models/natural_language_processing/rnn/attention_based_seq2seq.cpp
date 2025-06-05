#include "include/models/natural_language_processing/rnn/attention_based_seq2seq.h"


using namespace std;

namespace xt::models
{
    AttentionBasedSeq2Seq::AttentionBasedSeq2Seq(int num_classes, int in_channels)
    {
    }

    AttentionBasedSeq2Seq::AttentionBasedSeq2Seq(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void AttentionBasedSeq2Seq::reset()
    {
    }

    auto AttentionBasedSeq2Seq::forward(std::initializer_list<std::any> tensors) -> std::any
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
