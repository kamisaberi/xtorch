#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class SBU : torch::data::Dataset<SBU> {

    public :
       SBU();
    };
}
