#include <transforms/appliers/palindrome.h>

namespace xt::transforms
{
    Palindrome::Palindrome() = default;

    Palindrome::Palindrome(std::vector<xt::Module> transforms): xt::Module(), transforms(std::move(transforms))
    {
        throw std::runtime_error("Palindrome applier is not yet implemented.");
    }


    auto Palindrome::forward(std::initializer_list<std::any> tensors) -> std::any
    {
    //     std::vector<std::any> any_vec(tensors);
    //
    //     std::vector<torch::Tensor> tensor_vec;
    //     for (const auto& item : any_vec)
    //     {
    //         tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
    //     }
    //
    //     torch::Tensor input = tensor_vec[0];
    //
    //     for (const auto& transform : this->transforms)
    //     {
    //         input = std::any_cast<torch::Tensor>(transform({input}));
    //         // input = std::any_cast<torch::Tensor>(transforms({input}));
    //     }
    //     for (int i = this->transforms.size() - 1; i >= 0; i--)
    //     {
    //         input = std::any_cast<torch::Tensor>(transforms[i]({input}));
    //     }
    }

}
