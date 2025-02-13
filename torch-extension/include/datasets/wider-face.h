#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class WIDERFace : torch::data::Dataset<WIDERFace> {

    public :
       WIDERFace();
    };
}
