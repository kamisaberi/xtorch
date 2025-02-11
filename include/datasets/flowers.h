#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class Flowers102 : torch::data::Dataset<Flowers102> {

    public :
       Flowers102();
    };
}
