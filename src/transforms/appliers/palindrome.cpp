
#include "include/transforms/appliers/palindrome.h"

namespace xt::transforms
{

    Palindrome::Palindrome() = default;

    Palindrome::Palindrome(std::vector<xt::Module> transforms): xt::Module(), transforms(std::move(transforms))
    {
    }


    auto Palindrome::forward(std::initializer_list<torch::Tensor> tensors) -> std::any {

        std::vector<torch::Tensor> tensor_vec(tensors);
        input = tensor_vec[0];

        for (const auto& transform : this->transforms)
        {
            input = transform(std::move(input));
        }
        for (int i = this->transforms.size() - 1; i >= 0; i--)
        {
            input = transforms[i](std::move(input));
        }





    }

        torch::Tensor Palindrome::forward(torch::Tensor input) const
    {
        for (const auto& transform : this->transforms)
        {
            input = transform(std::move(input));
        }
        for (int i = this->transforms.size() - 1; i >= 0; i--)
        {
            input = transforms[i](std::move(input));
        }

        return input;

    }

}
