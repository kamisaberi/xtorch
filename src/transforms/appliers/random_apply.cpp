
#include <transforms/appliers/random_apply.h>

namespace xt::transforms
{

    RandomApply::RandomApply() = default;

    RandomApply::RandomApply(std::vector<xt::Module> transforms): xt::Module(), transforms(std::move(transforms))
    {
        throw std::runtime_error("RandomApply applier is not yet implemented.");
    }


    auto RandomApply::forward(std::initializer_list<std::any> tensors) -> std::any {

        // std::vector<torch::Tensor> tensor_vec(tensors);
        // torch::Tensor input = tensor_vec[0];
        //
        // int index = torch::randint(0, transforms.size() - 1, {}, torch::kInt32).item<int>();
        // input = std::any_cast<torch::Tensor>(transforms[index]({input}));
        // return input;

    }



    }
