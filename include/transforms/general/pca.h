#pragma once
#include "../common.h"

namespace xt::transforms::general {

    struct PCA final : xt::Module {
    public:
        PCA();
        PCA(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        std::function<torch::Tensor(torch::Tensor)> transform;
    };


}