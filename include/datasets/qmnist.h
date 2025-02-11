#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class QMNIST : torch::data::Dataset<QMNIST> {

    public :
        QMNIST();
    };
}
