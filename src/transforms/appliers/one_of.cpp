
#include <transforms/appliers/one_of.h>

namespace xt::transforms {

    OneOf::OneOf() = default;

    OneOf::OneOf(std::vector <xt::Module> transforms) : xt::Module(), transforms(std::move(transforms)) {
        throw std::runtime_error("OneOf applier is not yet implemented.");
    }


    auto OneOf::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor input = tensor_vec[0];
        int index = torch::randint(0, transforms.size() - 1, {}, torch::kInt32).item<int>();
        input = std::any_cast<torch::Tensor>(transforms[index]({input}));
        return input;
    }

}
