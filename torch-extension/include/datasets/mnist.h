#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class MNIST : torch::data::Dataset<MNIST> {

    public :
        MNIST();
    };
}
