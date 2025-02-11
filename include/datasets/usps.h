#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class USPS : torch::data::Dataset<USPS> {

    public :
       USPS();
    };
}
