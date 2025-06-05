#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class ElasticTransform final : public xt::Module
    {
    public:
        ElasticTransform();
        explicit ElasticTransform(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
