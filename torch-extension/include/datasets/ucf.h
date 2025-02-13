#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class UCF101 : public torch::data::Dataset<UCF101> {

    public :
       UCF101(const std::string  &root);
       torch::data::Example<> get(size_t index) override;
       torch::optional<size_t> size() const override;

    };
}
