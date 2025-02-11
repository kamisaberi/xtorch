#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class HD1K : torch::data::Dataset<HD1K> {

    public :
       HD1K();
    };
}
