#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor maxout(const torch::Tensor& x, int64_t num_pieces, int64_t dim = 1) ;

}



