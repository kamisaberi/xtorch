#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class EMNIST : torch::data::Dataset<EMNIST> {

    public :
       EMNIST();
    };
}
