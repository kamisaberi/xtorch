#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::target
{
    class MultiLabelBinarizer final : public xt::Module
    {
    public:
        MultiLabelBinarizer();
        explicit MultiLabelBinarizer(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
