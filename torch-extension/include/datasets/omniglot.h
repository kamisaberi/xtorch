#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class Omniglot : torch::data::Dataset<Omniglot> {

    public :
       Omniglot();
    };
}
