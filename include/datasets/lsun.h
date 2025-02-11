#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class LSUN : torch::data::Dataset<LSUN> {

    public :
       LSUN();
    };
}
