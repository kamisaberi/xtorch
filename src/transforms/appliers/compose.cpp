#include "include/transforms/appliers/compose.h"

namespace xt::transforms
{
    Compose::Compose() = default;

    Compose::Compose(std::vector<std::shared_ptr<xt::Module>> transforms): xt::Module(),
                                                                           transforms(std::move(transforms))
    {
    }

    auto Compose::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        // std::vector<std::any> any_vec(tensors);
        //
        // std::vector<torch::Tensor> tensor_vec;
        // for (const auto& item : any_vec)
        // {
        //     tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        // }
        //
        // torch::Tensor input = tensor_vec[0];
        // for (auto& transform : this->transforms)
        // {
        //
        //     input = std::any_cast<torch::Tensor>(transform({input}));
        // }
        // return input;

        if (tensors.size() == 0)
        {
            throw std::runtime_error("Compose::forward expects at least one input.");
        }
        std::any current_data = *tensors.begin(); // Start with the first input
        for (const auto& transform_module : transforms) // Iterate over shared_ptrs
        {
            if (transform_module) // Check if the pointer is valid
            {
                current_data = transform_module->forward({current_data});
            }
            else
            {
            }
        }
        return current_data;
    }
}
