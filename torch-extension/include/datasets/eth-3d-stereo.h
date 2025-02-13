#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class ETH3DStereo : torch::data::Dataset<ETH3DStereo> {

    public :
       ETH3DStereo();
    };
}
