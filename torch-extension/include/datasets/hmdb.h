#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class HMDB51 : torch::data::Dataset<HMDB51> {

    public :
       HMDB51();
    };
}
