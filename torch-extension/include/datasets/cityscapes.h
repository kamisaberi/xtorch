#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class Cityscapes : torch::data::Dataset<Cityscapes> {

    public :
       Cityscapes();
    };
}
