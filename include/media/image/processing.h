#pragma once

#include <iostream>
#include <torch/torch.h>

namespace torch {
    namespace ext {
        namespace media {
            namespace image {
                torch::Tensor resize(const torch::Tensor &tensor, const std::vector<int64_t> &size);
            }
        }
    }
}

