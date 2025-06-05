//TODO SHOULD CHANGE
#pragma once
#include "../common.h"


namespace xt::transforms::image
{
    struct Resize final : public xt::Module
    {
    public:
        explicit Resize(std::vector<int64_t> size);

        // torch::Tensor operator()(torch::Tensor img);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
    private:
        std::vector<int64_t> size;
    };
}
