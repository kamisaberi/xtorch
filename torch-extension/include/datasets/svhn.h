#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class SVHN : torch::data::Dataset<SVHN> {

    public :
       SVHN();
    };
}
