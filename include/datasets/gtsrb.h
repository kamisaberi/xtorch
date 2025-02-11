#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class GTSRB : torch::data::Dataset<GTSRB> {

    public :
       GTSRB();
    };
}
