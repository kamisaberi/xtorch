#pragma once
#include "transforms/common.h"
namespace xt::transforms::general {

    struct PCA final : xt::Module {
    public:
        PCA();
        PCA(std::function<torch::Tensor(torch::Tensor)> transform);
        torch::Tensor forward(torch::Tensor input) const override;

    private:
        std::function<torch::Tensor(torch::Tensor)> transform;
    };


}