#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class Compose final : public xt::Module
    {
    public:
        Compose();
        explicit Compose(std::vector<std::shared_ptr<xt::Module>> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        std::vector<std::shared_ptr<xt::Module>> transforms; // Store shared_ptr
        // std::vector<xt::Module> transforms;
    };
}
