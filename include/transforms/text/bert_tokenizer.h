#pragma once

#include "../common.h"


namespace xt::transforms::text
{
    class BertTokenizer final : public xt::Module
    {
    public:
        BertTokenizer();
        explicit BertTokenizer(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
