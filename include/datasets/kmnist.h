#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class KMNIST : torch::data::Dataset<KMNIST> {

    public :
       KMNIST();
    };
}
