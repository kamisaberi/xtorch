#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class RenderedSST2 : torch::data::Dataset<RenderedSST2> {

    public :
       RenderedSST2();
    };
}
