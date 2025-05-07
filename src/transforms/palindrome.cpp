#include <utility>

#include "../../include/transforms/palindrome.h"

namespace xt::transforms
{
    using TransformFunc = std::function<torch::Tensor(torch::Tensor)>;

    Palindrome::Palindrome() = default;

    Palindrome::Palindrome(std::vector<TransformFunc> transforms): xt::Module(), transforms(std::move(transforms))
    {
    }

    torch::Tensor Palindrome::operator()(torch::Tensor input) const
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
