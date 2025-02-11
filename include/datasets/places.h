#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class Places365 : torch::data::Dataset<Places365> {

    public :
       Places365();
    };
}
