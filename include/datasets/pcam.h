#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class PCAM : torch::data::Dataset<PCAM> {

    public :
       PCAM();
    };
}
